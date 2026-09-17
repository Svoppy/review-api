from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_variant(raw_value: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in raw_value.split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    if "label" not in values:
        raise ValueError(f"Variant '{raw_value}' must include label=...")
    return values


def apply_override(payload: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    current = payload
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    if raw_value.lower() in {"true", "false"}:
        value: Any = raw_value.lower() == "true"
    else:
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
    current[keys[-1]] = value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed experiment matrix for ablations such as loss weights, backbones, and data scales."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[11, 21, 42, 84, 126],
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help=(
            "Comma-separated overrides. Example: "
            "label=xlmr,model_name=FacebookAI/xlm-roberta-base,loss_weights.sentiment=0.3"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "single-task-sentiment", "single-task-authenticity", "multitask"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8")) or {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    for raw_variant in args.variant:
        variant = parse_variant(raw_variant)
        label = variant.pop("label")
        input_path = Path(variant.pop("input", str(args.input)))

        payload = yaml.safe_load(yaml.safe_dump(base_config)) or {}
        for key, value in variant.items():
            apply_override(payload, key, value)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{label}.yaml",
            delete=False,
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
            config_path = Path(handle.name)

        command = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "run_multiseed_experiments.py"),
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output-root",
            str(args.output_root / label),
            "--report-dir",
            str(args.report_root / label),
            "--split-seed",
            str(args.split_seed),
            "--models",
            *args.models,
            "--train-seeds",
            *[str(seed) for seed in args.train_seeds],
        ]
        if args.skip_existing:
            command.append("--skip-existing")

        print("[matrix]", label)
        print("         " + " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
