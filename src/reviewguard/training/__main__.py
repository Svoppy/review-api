from __future__ import annotations

import argparse
from pathlib import Path

from reviewguard.data import load_unified_records
from reviewguard.training.baseline import BaselineConfig, ClassicalBaselineTrainer
from reviewguard.training.export import ensure_export_dir, write_json
from reviewguard.training.splits import split_unified_records


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
    parser.add_argument("--random-state", type=int, default=42)


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

    return parser


def _write_report(export_dir: Path, payload: dict) -> None:
    write_json(export_dir / "train_report.json", payload)


def run_baseline(args: argparse.Namespace) -> int:
    records = load_unified_records(args.input_path)
    split = split_unified_records(
        records,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    trainer = ClassicalBaselineTrainer(BaselineConfig(random_state=args.random_state))
    trainer.fit(split["train"])
    export_dir = trainer.export(args.export_dir)

    report = {
        "trainer": "baseline",
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "validation_metrics": trainer.evaluate(split["valid"]) if split["valid"] else {},
        "test_metrics": trainer.evaluate(split["test"]) if split["test"] else {},
    }
    _write_report(export_dir, report)
    return 0


def run_multitask(args: argparse.Namespace) -> int:
    from reviewguard.training.multitask import MultitaskTrainingConfig, MultitaskTrainingScaffold

    records = load_unified_records(args.input_path)
    split = split_unified_records(
        records,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    raw_config = _read_model_config(args.config_path)
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
    )

    trainer = MultitaskTrainingScaffold(config)
    fit_summary = trainer.fit(split["train"], valid_records=split["valid"] or None)
    export_dir = trainer.export(args.export_dir)

    report = {
        "trainer": "multitask",
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
    if args.command == "multitask":
        return run_multitask(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
