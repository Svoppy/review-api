from __future__ import annotations

import hashlib
import json
import re
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


def _text_fingerprint(record: Mapping[str, Any]) -> str | None:
    text = _normalized_text(record)
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if not normalized:
        return None
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


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
        "minority_label": min(counts, key=counts.get) if labeled else None,
        "minimum_class_support": min(counts.values()) if labeled else 0,
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
    fingerprint_sets = {
        split_name: {
            fingerprint
            for record in split.get(split_name, [])
            if (fingerprint := _text_fingerprint(record)) is not None
        }
        for split_name in split_names
    }

    overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        pair_key = f"{left}_vs_{right}"
        overlaps[pair_key] = {
            "exact_text_overlap": len(text_sets[left] & text_sets[right]),
            "record_id_overlap": len(id_sets[left] & id_sets[right]),
            "normalized_text_overlap": len(fingerprint_sets[left] & fingerprint_sets[right]),
        }
    return overlaps


def _field_distribution(records: Iterable[Mapping[str, Any]], *, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _task_group_coverage(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    group_field: str,
) -> dict[str, dict[str, Any]]:
    label_field = TASK_FIELDS[task]
    labels = TASK_LABELS[task]
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        group_value = str(record.get(group_field) or "unknown")
        payload = grouped.setdefault(
            group_value,
            {
                "records": 0,
                "labeled_records": 0,
                "counts": {label: 0 for label in labels},
            },
        )
        payload["records"] += 1
        label = record.get(label_field)
        if label is None:
            continue
        payload["labeled_records"] += 1
        payload["counts"][str(label)] = payload["counts"].get(str(label), 0) + 1

    for payload in grouped.values():
        labeled_records = int(payload["labeled_records"])
        payload["label_coverage"] = (
            labeled_records / int(payload["records"]) if payload["records"] else 0.0
        )
    return dict(sorted(grouped.items()))


def _label_source_coverage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": {
            task: _task_group_coverage(records, task=task, group_field="source")
            for task in TASK_FIELDS
        },
        "language": {
            task: _task_group_coverage(records, task=task, group_field="language")
            for task in TASK_FIELDS
        },
        "domain": {
            task: _task_group_coverage(records, task=task, group_field="domain")
            for task in TASK_FIELDS
        },
    }


def _duplicate_summary(records: list[Mapping[str, Any]]) -> dict[str, int]:
    exact_counts: dict[str, int] = {}
    normalized_counts: dict[str, int] = {}
    for record in records:
        exact_text = _normalized_text(record)
        if exact_text:
            exact_counts[exact_text] = exact_counts.get(exact_text, 0) + 1
        fingerprint = _text_fingerprint(record)
        if fingerprint is not None:
            normalized_counts[fingerprint] = normalized_counts.get(fingerprint, 0) + 1

    exact_duplicate_rows = sum(count - 1 for count in exact_counts.values() if count > 1)
    normalized_duplicate_rows = sum(count - 1 for count in normalized_counts.values() if count > 1)
    return {
        "exact_duplicate_rows": exact_duplicate_rows,
        "normalized_duplicate_rows": normalized_duplicate_rows,
        "unique_exact_texts": len(exact_counts),
        "unique_normalized_texts": len(normalized_counts),
    }


def _task_warnings(
    split: Mapping[str, list[Mapping[str, Any]]],
    *,
    task: str,
    minimum_reliable_class_support: int = 20,
    minimum_reliable_eval_support: int = 100,
) -> list[str]:
    warnings: list[str] = []
    train_distribution = _task_distribution(split.get("train", []), task=task)
    test_distribution = _task_distribution(split.get("test", []), task=task)

    if test_distribution["labeled_records"] < minimum_reliable_eval_support:
        warnings.append(
            f"{task}: test split has only {test_distribution['labeled_records']} labeled examples; "
            "reported metrics should be treated as pilot-scale and high-variance."
        )

    min_class_support = int(test_distribution["minimum_class_support"])
    minority_label = test_distribution["minority_label"]
    if test_distribution["labeled_records"] and min_class_support < minimum_reliable_class_support:
        warnings.append(
            f"{task}: minority class '{minority_label}' has only {min_class_support} test examples; "
            "macro-F1 is likely unstable."
        )

    if train_distribution["labeled_records"] and train_distribution["minimum_class_support"] < minimum_reliable_class_support:
        warnings.append(
            f"{task}: training split has sparse class coverage (minimum class support "
            f"{train_distribution['minimum_class_support']}); class balancing or more data is recommended."
        )

    return warnings


def _global_warnings(records: list[Mapping[str, Any]], split: Mapping[str, list[Mapping[str, Any]]]) -> list[str]:
    warnings: list[str] = []
    duplicate_summary = _duplicate_summary(records)
    overlap = _overlap_counts(split)

    if duplicate_summary["normalized_duplicate_rows"] > 0:
        warnings.append(
            f"Detected {duplicate_summary['normalized_duplicate_rows']} normalized duplicate rows in the merged corpus; "
            "group-aware splitting is recommended."
        )

    if any(pair["normalized_text_overlap"] > 0 for pair in overlap.values()):
        warnings.append("Normalized text overlap exists across splits, indicating possible leakage.")

    domain_counts = _field_distribution(records, field="domain")
    language_counts = _field_distribution(records, field="language")
    if len(domain_counts) > 1:
        warnings.append(
            "Corpus mixes multiple domains; cross-domain robustness should be reported separately from aggregate scores."
        )
    if len(language_counts) > 1:
        warnings.append(
            "Corpus mixes multiple languages; language-aware evaluation is required for strong claims."
        )

    return warnings


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
        "duplicate_summary": _duplicate_summary(records),
        "source_distribution": _field_distribution(records, field="source"),
        "domain_distribution": _field_distribution(records, field="domain"),
        "language_distribution": _field_distribution(records, field="language"),
        "label_coverage": _label_source_coverage(records),
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "split_overlap": _overlap_counts(split),
        "tasks": {},
    }

    for task in TASK_FIELDS:
        report["tasks"][task] = {
            "train_distribution": _task_distribution(split.get("train", []), task=task),
            "validation_distribution": _task_distribution(split.get("valid", []), task=task),
            "test_distribution": _task_distribution(split.get("test", []), task=task),
            "warnings": _task_warnings(split, task=task),
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

    report["warnings"] = _global_warnings(records, split)
    return report


def write_audit_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
