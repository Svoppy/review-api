"""Validate an article-grade training run before any model weights are written.

The report produced by this script is a provenance record, not experimental evidence.
It binds a particular corpus, split, configuration, and repository state so that later
model results cannot be accidentally associated with a different local snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.data import build_dataset_audit_report, load_unified_records
from reviewguard.training.runtime import resolve_training_device
from reviewguard.training.splits import split_unified_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "changed_paths": [] if not status else status.splitlines(),
    }


def task_source_support(records: list[dict[str, Any]], label_field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(record.get("source") or "unknown")
                for record in records
                if record.get(label_field) is not None
            ).items()
        )
    )


def overlap_is_zero(audit: dict[str, Any]) -> bool:
    overlap = audit.get("split_overlap", {})
    return all(
        int(values.get(metric, 0)) == 0
        for values in overlap.values()
        for metric in ("exact_text_overlap", "normalized_text_overlap", "record_id_overlap")
    )


def training_runtime_status(config: dict[str, Any]) -> dict[str, Any]:
    requested = str(config.get("train", {}).get("device", "auto"))
    requires_accelerator = bool(config.get("experiment", {}).get("require_accelerator", False))
    try:
        device = resolve_training_device(requested)
    except (RuntimeError, ValueError) as error:
        return {
            "requested": requested,
            "effective": None,
            "accelerator_available": False,
            "requires_accelerator": requires_accelerator,
            "resolution_error": str(error),
        }
    return {
        "requested": device.requested,
        "effective": device.resolved,
        "accelerator_available": device.accelerator_available,
        "requires_accelerator": requires_accelerator,
        "resolution_error": None,
    }


def build_preflight_report(
    *,
    input_path: Path,
    config_path: Path,
    split_seed: int,
    min_sentiment_class_support: int,
    min_authenticity_class_support: int,
) -> dict[str, Any]:
    records = load_unified_records(input_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training_runtime = training_runtime_status(config)
    split = split_unified_records(records, random_state=split_seed)
    audit = build_dataset_audit_report(
        records,
        split,
        input_path=input_path,
        random_state=split_seed,
    )

    sentiment_test = audit["tasks"]["sentiment"]["test_distribution"]
    authenticity_test = audit["tasks"]["authenticity"]["test_distribution"]
    sentiment_source_support = task_source_support(split["test"], "sentiment_label")
    authenticity_source_support = task_source_support(split["test"], "authenticity_label")

    data_checks = {
        "zero_cross_split_overlap": overlap_is_zero(audit),
        "sentiment_minimum_class_support": int(sentiment_test["minimum_class_support"])
        >= min_sentiment_class_support,
        "authenticity_minimum_class_support": int(authenticity_test["minimum_class_support"])
        >= min_authenticity_class_support,
        "multitask_architecture_explicit": bool(config.get("pooling"))
        and bool(config.get("head_type")),
        "model_name_explicit": bool(config.get("model_name")),
    }
    execution_checks = {
        "device_configuration_valid": training_runtime["resolution_error"] is None,
        "accelerator_requirement_satisfied": (
            not training_runtime["requires_accelerator"]
            or training_runtime["accelerator_available"]
        ),
    }
    warnings = list(audit.get("warnings", []))
    if len(authenticity_source_support) < 2:
        warnings.append(
            "Authenticity labels in the test split come from fewer than two sources; "
            "this run can support a controlled single-source authenticity claim, not a "
            "multi-benchmark generalization claim."
        )
    if not execution_checks["accelerator_requirement_satisfied"]:
        warnings.append(
            "The selected article configuration requires CUDA or Apple Metal, but no supported "
            "accelerator is available in this runtime."
        )

    return {
        "kind": "article_training_preflight",
        "data_ready_for_training": all(data_checks.values()),
        "ready_to_train": all(data_checks.values()) and all(execution_checks.values()),
        "claim_scope": (
            "controlled mixed-source study with single-source authenticity evidence"
            if len(authenticity_source_support) < 2
            else "multi-source authenticity evaluation"
        ),
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "records": len(records),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256(config_path),
            "model_name": config.get("model_name"),
            "pooling": config.get("pooling"),
            "head_type": config.get("head_type"),
        },
        "split_seed": split_seed,
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "test_support": {
            "sentiment": {
                "class_counts": sentiment_test["counts"],
                "minimum_class_support": sentiment_test["minimum_class_support"],
                "source_counts": sentiment_source_support,
            },
            "authenticity": {
                "class_counts": authenticity_test["counts"],
                "minimum_class_support": authenticity_test["minimum_class_support"],
                "source_counts": authenticity_source_support,
            },
        },
        "checks": data_checks,
        "execution_checks": execution_checks,
        "warnings": warnings,
        "git": git_state(PROJECT_ROOT),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "training_device": training_runtime,
        },
        "audit": audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a provenance and readiness report for an article-grade experiment."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-sentiment-class-support", type=int, default=100)
    parser.add_argument("--min-authenticity-class-support", type=int, default=250)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit unsuccessfully when a required readiness check fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_preflight_report(
        input_path=args.input,
        config_path=args.config,
        split_seed=args.split_seed,
        min_sentiment_class_support=args.min_sentiment_class_support,
        min_authenticity_class_support=args.min_authenticity_class_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote preflight report to {args.output}")
    print(f"ready_to_train={report['ready_to_train']}")
    print(f"claim_scope={report['claim_scope']}")
    if args.strict and not report["ready_to_train"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
