from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from reviewguard.data.pipeline import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.training.metrics import compute_classification_metrics


TASK_FIELDS = {
    "sentiment": "sentiment_label",
    "authenticity": "authenticity_label",
}

TASK_LABELS = {
    "sentiment": list(SENTIMENT_LABELS),
    "authenticity": list(AUTHENTICITY_LABELS),
}


def hash_input_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(record: Mapping[str, Any]) -> str:
    return str(record.get("text") or "").strip()


def _record_identifier(record: Mapping[str, Any]) -> str | None:
    record_id = record.get("record_id")
    if record_id in (None, ""):
        return None
    source = str(record.get("source") or "unknown")
    return f"{source}:{record_id}"


def _task_distribution(records: Iterable[Mapping[str, Any]], *, task: str) -> dict[str, Any]:
    label_field = TASK_FIELDS[task]
    labels = TASK_LABELS[task]
    counts = {label: 0 for label in labels}
    labeled = 0

    for record in records:
        label = record.get(label_field)
        if label is None:
            continue
        labeled += 1
        counts[str(label)] = counts.get(str(label), 0) + 1

    proportions = {
        label: (count / labeled if labeled else 0.0) for label, count in counts.items()
    }
    majority_label = max(counts, key=counts.get) if labeled else None
    return {
        "labeled_records": labeled,
        "counts": counts,
        "proportions": proportions,
        "majority_label": majority_label,
    }


def _majority_baseline_metrics(
    train_records: list[Mapping[str, Any]],
    eval_records: list[Mapping[str, Any]],
    *,
    task: str,
) -> dict[str, Any]:
    train_distribution = _task_distribution(train_records, task=task)
    majority_label = train_distribution["majority_label"]
    if majority_label is None:
        return {}

    label_field = TASK_FIELDS[task]
    labels = TASK_LABELS[task]
    y_true: list[str] = []
    y_pred: list[str] = []
    for record in eval_records:
        label = record.get(label_field)
        if label is None:
            continue
        y_true.append(str(label))
        y_pred.append(str(majority_label))

    if not y_true:
        return {}

    metrics = compute_classification_metrics(y_true, y_pred, labels=labels).to_dict()
    metrics["majority_label"] = majority_label
    return metrics


def _overlap_counts(split: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    split_names = ("train", "valid", "test")
    text_sets: dict[str, set[str]] = {}
    id_sets: dict[str, set[str]] = {}

    for split_name in split_names:
        records = split.get(split_name, [])
        text_sets[split_name] = {_normalized_text(record) for record in records if _normalized_text(record)}
        id_sets[split_name] = {
            identifier
            for record in records
            if (identifier := _record_identifier(record)) is not None
        }

    overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        pair_key = f"{left}_vs_{right}"
        overlaps[pair_key] = {
            "exact_text_overlap": len(text_sets[left] & text_sets[right]),
            "record_id_overlap": len(id_sets[left] & id_sets[right]),
        }
    return overlaps


def build_dataset_audit_report(
    records: list[Mapping[str, Any]],
    split: Mapping[str, list[Mapping[str, Any]]],
    *,
    input_path: str | Path | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "input_path": str(input_path) if input_path is not None else None,
        "input_sha256": hash_input_file(input_path) if input_path is not None else None,
        "random_state": random_state,
        "records": len(records),
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "split_overlap": _overlap_counts(split),
        "tasks": {},
    }

    for task in TASK_FIELDS:
        report["tasks"][task] = {
            "train_distribution": _task_distribution(split.get("train", []), task=task),
            "validation_distribution": _task_distribution(split.get("valid", []), task=task),
            "test_distribution": _task_distribution(split.get("test", []), task=task),
            "majority_baseline": {
                "validation": _majority_baseline_metrics(
                    split.get("train", []),
                    split.get("valid", []),
                    task=task,
                ),
                "test": _majority_baseline_metrics(
                    split.get("train", []),
                    split.get("test", []),
                    task=task,
                ),
            },
        }

    return report


def write_audit_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
