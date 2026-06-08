from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from reviewguard.data import load_unified_records
from reviewguard.training.baseline import BaselineConfig, ClassicalBaselineTrainer
from reviewguard.training.export import ensure_export_dir, write_json
from reviewguard.training.splits import split_unified_records


TASK_CHOICES = ("sentiment", "authenticity")


def _read_model_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _add_shared_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--export-dir", required=True, dest="export_dir")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ReviewGuard baselines or multitask models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Train TF-IDF + LogisticRegression baselines.")
    _add_shared_split_args(baseline)

    multitask = subparsers.add_parser("multitask", help="Train the multitask Transformer scaffold.")
    _add_shared_split_args(multitask)
    multitask.add_argument(
        "--config",
        default="configs/model.multitask.yaml",
        dest="config_path",
        help="Path to YAML config for the multitask trainer.",
    )

    single_task = subparsers.add_parser(
        "single-task",
        help="Train a single-task Transformer baseline for sentiment or authenticity.",
    )
    _add_shared_split_args(single_task)
    single_task.add_argument(
        "--task",
        required=True,
        choices=TASK_CHOICES,
        help="Which task-specific label family to train.",
    )
    single_task.add_argument(
        "--config",
        default="configs/model.multitask.yaml",
        dest="config_path",
        help="Path to YAML config for the single-task trainer.",
    )

    return parser


def _write_report(export_dir: Path, payload: dict) -> None:
    write_json(export_dir / "train_report.json", payload)


def _collect_runtime_metadata() -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        PackageNotFoundError = Exception  # type: ignore[assignment]
        version = None  # type: ignore[assignment]

    for package_name in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "fastapi"):
        if version is None:
            package_versions[package_name] = None
            continue
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def _hash_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_input_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    sentiment_labeled = 0
    authenticity_labeled = 0
    for record in records:
        source = str(record.get("source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        if record.get("sentiment_label") is not None:
            sentiment_labeled += 1
        if record.get("authenticity_label") is not None:
            authenticity_labeled += 1
    return {
        "records": len(records),
        "sources": sources,
        "sentiment_labeled": sentiment_labeled,
        "authenticity_labeled": authenticity_labeled,
    }


def _pick_config_value(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_split_random_state(
    cli_random_state: int | None,
    raw_config: dict | None = None,
    *,
    task: str | None = None,
) -> int:
    if cli_random_state is not None:
        return cli_random_state

    raw_config = raw_config or {}
    if task is not None:
        task_cfg = raw_config.get("single_task", {}).get("tasks", {}).get(task, {})
        single_task_train_cfg = raw_config.get("single_task", {}).get("train", {})
        configured = _pick_config_value(
            task_cfg.get("random_state"),
            single_task_train_cfg.get("random_state"),
            raw_config.get("train", {}).get("random_state"),
            default=None,
        )
        if configured is not None:
            return int(configured)

    configured = raw_config.get("train", {}).get("random_state")
    if configured is not None:
        return int(configured)
    return 42


def _resolve_single_task_config(raw_config: dict, task: str, *, random_state: int | None) -> Any:
    from reviewguard.training.single_task_config import (
        TASK_LABELS,
        SingleTaskTrainingConfig,
    )

    single_task_cfg = raw_config.get("single_task", {})
    shared_train_cfg = raw_config.get("train", {})
    single_task_train_cfg = single_task_cfg.get("train", {})
    task_cfg = single_task_cfg.get("tasks", {}).get(task, {})

    task_labels = task_cfg.get("labels")
    if not task_labels:
        task_labels = raw_config.get(
            "sentiment_labels" if task == "sentiment" else "authenticity_labels",
            TASK_LABELS[task],
        )

    return SingleTaskTrainingConfig(
        task=task,
        labels=list(task_labels),
        model_name=_pick_config_value(
            task_cfg.get("model_name"),
            raw_config.get("model_name"),
            default=SingleTaskTrainingConfig.model_name,
        ),
        max_length=_pick_config_value(
            task_cfg.get("max_length"),
            raw_config.get("max_length"),
            default=SingleTaskTrainingConfig.max_length,
        ),
        batch_size=_pick_config_value(
            task_cfg.get("batch_size"),
            single_task_train_cfg.get("batch_size"),
            shared_train_cfg.get("batch_size"),
            default=SingleTaskTrainingConfig.batch_size,
        ),
        learning_rate=_pick_config_value(
            task_cfg.get("learning_rate"),
            single_task_train_cfg.get("learning_rate"),
            shared_train_cfg.get("learning_rate"),
            default=SingleTaskTrainingConfig.learning_rate,
        ),
        weight_decay=_pick_config_value(
            task_cfg.get("weight_decay"),
            single_task_train_cfg.get("weight_decay"),
            shared_train_cfg.get("weight_decay"),
            default=SingleTaskTrainingConfig.weight_decay,
        ),
        epochs=_pick_config_value(
            task_cfg.get("epochs"),
            single_task_train_cfg.get("epochs"),
            shared_train_cfg.get("epochs"),
            default=SingleTaskTrainingConfig.epochs,
        ),
        dropout=_pick_config_value(
            task_cfg.get("dropout"),
            single_task_cfg.get("dropout"),
            raw_config.get("dropout"),
            default=SingleTaskTrainingConfig.dropout,
        ),
        device=_pick_config_value(
            task_cfg.get("device"),
            single_task_train_cfg.get("device"),
            shared_train_cfg.get("device"),
            default=SingleTaskTrainingConfig.device,
        ),
        random_state=_pick_config_value(
            random_state,
            task_cfg.get("random_state"),
            single_task_train_cfg.get("random_state"),
            shared_train_cfg.get("random_state"),
            default=SingleTaskTrainingConfig.random_state,
        ),
    )


def run_baseline(args: argparse.Namespace) -> int:
    records = load_unified_records(args.input_path)
    split_random_state = _resolve_split_random_state(args.random_state)
    split = split_unified_records(
        records,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=split_random_state,
    )
    trainer = ClassicalBaselineTrainer(BaselineConfig(random_state=split_random_state))
    trainer.fit(split["train"])
    export_dir = trainer.export(args.export_dir)

    report = {
        "trainer": "baseline",
        "input_path": str(args.input_path),
        "input_sha256": _hash_file(args.input_path),
        "input_summary": _summarize_input_records(records),
        "runtime": _collect_runtime_metadata(),
        "random_state": split_random_state,
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "validation_metrics": trainer.evaluate(split["valid"]) if split["valid"] else {},
        "test_metrics": trainer.evaluate(split["test"]) if split["test"] else {},
    }
    _write_report(export_dir, report)
    return 0


def run_single_task(args: argparse.Namespace) -> int:
    from reviewguard.training.single_task import (
        SingleTaskTransformerTrainer,
        labeled_records_for_task,
    )

    raw_config = _read_model_config(args.config_path)
    records = load_unified_records(args.input_path)
    split_random_state = _resolve_split_random_state(
        args.random_state,
        raw_config,
        task=args.task,
    )
    split = split_unified_records(
        records,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=split_random_state,
    )

    trainer = SingleTaskTransformerTrainer(
        _resolve_single_task_config(raw_config, args.task, random_state=args.random_state)
    )
    fit_summary = trainer.fit(split["train"], valid_records=split["valid"] or None)
    export_dir = trainer.export(args.export_dir)

    report = {
        "trainer": "single_task_transformer",
        "task": args.task,
        "input_path": str(args.input_path),
        "input_sha256": _hash_file(args.input_path),
        "input_summary": _summarize_input_records(records),
        "runtime": _collect_runtime_metadata(),
        "random_state": split_random_state,
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "labeled_split_sizes": {
            name: len(labeled_records_for_task(rows, args.task)) for name, rows in split.items()
        },
        "fit_summary": fit_summary,
        "validation_metrics": trainer.evaluate(split["valid"]) if split["valid"] else {},
        "test_metrics": trainer.evaluate(split["test"]) if split["test"] else {},
    }
    _write_report(export_dir, report)
    return 0


def run_multitask(args: argparse.Namespace) -> int:
    from reviewguard.training.multitask import MultitaskTrainingConfig, MultitaskTrainingScaffold

    raw_config = _read_model_config(args.config_path)
    records = load_unified_records(args.input_path)
    split_random_state = _resolve_split_random_state(args.random_state, raw_config)
    split = split_unified_records(
        records,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=split_random_state,
    )

    train_cfg = raw_config.get("train", {})
    loss_weights = raw_config.get("loss_weights", {})
    config = MultitaskTrainingConfig(
        model_name=raw_config.get("model_name", MultitaskTrainingConfig.model_name),
        max_length=raw_config.get("max_length", MultitaskTrainingConfig.max_length),
        batch_size=train_cfg.get("batch_size", MultitaskTrainingConfig.batch_size),
        learning_rate=train_cfg.get("learning_rate", MultitaskTrainingConfig.learning_rate),
        weight_decay=train_cfg.get("weight_decay", MultitaskTrainingConfig.weight_decay),
        epochs=train_cfg.get("epochs", MultitaskTrainingConfig.epochs),
        dropout=raw_config.get("dropout", MultitaskTrainingConfig.dropout),
        sentiment_loss_weight=loss_weights.get(
            "sentiment",
            MultitaskTrainingConfig.sentiment_loss_weight,
        ),
        authenticity_loss_weight=loss_weights.get(
            "authenticity",
            MultitaskTrainingConfig.authenticity_loss_weight,
        ),
        device=train_cfg.get("device", MultitaskTrainingConfig.device),
        random_state=split_random_state,
    )

    trainer = MultitaskTrainingScaffold(config)
    fit_summary = trainer.fit(split["train"], valid_records=split["valid"] or None)
    export_dir = trainer.export(args.export_dir)

    report = {
        "trainer": "multitask",
        "input_path": str(args.input_path),
        "input_sha256": _hash_file(args.input_path),
        "input_summary": _summarize_input_records(records),
        "runtime": _collect_runtime_metadata(),
        "random_state": split_random_state,
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "fit_summary": fit_summary,
        "validation_metrics": trainer.evaluate(split["valid"]) if split["valid"] else {},
        "test_metrics": trainer.evaluate(split["test"]) if split["test"] else {},
    }
    _write_report(export_dir, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ensure_export_dir(args.export_dir)

    if args.command == "baseline":
        return run_baseline(args)
    if args.command == "single-task":
        return run_single_task(args)
    if args.command == "multitask":
        return run_multitask(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
