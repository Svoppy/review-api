"""Run the complete, provenance-bound article experiment on an accelerator.

The script deliberately fails before writing model artifacts when the selected
article configuration requires an accelerator that is not available.  It also
freezes the YAML configuration used for the run and records every child command
in a release manifest, so the numerical results can be traced back to one input
corpus and one explicit protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.training.runtime import require_accelerator, resolve_training_device


DEFAULT_TRAIN_SEEDS = [11, 21, 42, 84, 126]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_train_seeds(config: dict[str, Any], cli_seeds: list[int] | None) -> list[int]:
    if cli_seeds is not None:
        seeds = cli_seeds
    else:
        seeds = config.get("experiment", {}).get("train_seeds", DEFAULT_TRAIN_SEEDS)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.train_seeds must be a non-empty list of integers.")
    normalized = [int(seed) for seed in seeds]
    if len(set(normalized)) != len(normalized):
        raise ValueError("train seeds must be unique.")
    return normalized


def build_stage_commands(
    *,
    python_executable: str,
    input_path: Path,
    config_snapshot: Path,
    preflight_report: Path,
    output_root: Path,
    report_root: Path,
    statistics_root: Path,
    ablation_root: Path,
    split_seed: int,
    train_seeds: list[int],
    bootstrap_samples: int,
    randomization_samples: int,
    skip_existing: bool,
) -> list[tuple[str, list[str]]]:
    commands = [
        (
            "preflight",
            [
                python_executable,
                str(PROJECT_ROOT / "scripts" / "preflight_article_experiment.py"),
                "--input",
                str(input_path),
                "--config",
                str(config_snapshot),
                "--output",
                str(preflight_report),
                "--split-seed",
                str(split_seed),
                "--strict",
            ],
        ),
        (
            "multiseed_training",
            [
                python_executable,
                str(PROJECT_ROOT / "scripts" / "run_multiseed_experiments.py"),
                "--input",
                str(input_path),
                "--config",
                str(config_snapshot),
                "--output-root",
                str(output_root),
                "--report-dir",
                str(report_root),
                "--split-seed",
                str(split_seed),
                "--train-seeds",
                *[str(seed) for seed in train_seeds],
                "--python",
                python_executable,
            ],
        ),
        (
            "statistical_analysis",
            [
                python_executable,
                str(PROJECT_ROOT / "scripts" / "analyze_multiseed_statistics.py"),
                "--input",
                str(input_path),
                "--multiseed-root",
                str(output_root),
                "--summary-json",
                str(report_root / "summary.json"),
                "--output-dir",
                str(statistics_root),
                "--split-seed",
                str(split_seed),
                "--train-seeds",
                *[str(seed) for seed in train_seeds],
                "--bootstrap-samples",
                str(bootstrap_samples),
                "--randomization-samples",
                str(randomization_samples),
                "--device",
                "auto",
            ],
        ),
        (
            "task_ablation",
            [
                python_executable,
                str(PROJECT_ROOT / "scripts" / "build_task_ablation_report.py"),
                "--multiseed-summary",
                str(report_root / "summary.json"),
                "--statistics-summary",
                str(statistics_root / "statistics_summary.json"),
                "--output-dir",
                str(ablation_root),
            ],
        ),
    ]
    if skip_existing:
        commands[1][1].append("--skip-existing")
    return commands


def snapshot_config(source: Path, destination: Path) -> str:
    payload = source.read_text(encoding="utf-8")
    if destination.exists() and destination.read_text(encoding="utf-8") != payload:
        raise RuntimeError(
            f"Refusing to overwrite a different configuration snapshot at {destination}. "
            "Choose a new --report-root for a new protocol."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return sha256(destination)


def child_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(SRC_ROOT), existing_pythonpath))
    )
    environment["REVIEWGUARD_DEVICE"] = device
    return environment


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full article experiment: preflight, multi-seed training, statistics, ablation."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/joint_reviews.article20k.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.article.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("models/multiseed/article20k"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/multiseed/article20k"))
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("reports/preflight/article20k_preflight.json"),
    )
    parser.add_argument(
        "--statistics-root",
        type=Path,
        default=Path("reports/multiseed/article20k_statistics"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("reports/multiseed/article20k_ablation"),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seeds", nargs="+", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--randomization-samples", type=int, default=2000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable stage plan without requiring a GPU or writing artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input corpus was not found: {args.input}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Model configuration was not found: {args.config}")

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    train_seeds = resolve_train_seeds(raw_config, args.train_seeds)
    config_snapshot = args.report_root / "config.used.yaml"
    commands = build_stage_commands(
        python_executable=args.python,
        input_path=args.input,
        config_snapshot=config_snapshot,
        preflight_report=args.preflight_report,
        output_root=args.output_root,
        report_root=args.report_root,
        statistics_root=args.statistics_root,
        ablation_root=args.ablation_root,
        split_seed=args.split_seed,
        train_seeds=train_seeds,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        skip_existing=args.skip_existing,
    )

    if args.dry_run:
        print(json.dumps({"train_seeds": train_seeds, "stages": commands}, ensure_ascii=False, indent=2))
        return 0

    requested_device = str(raw_config.get("train", {}).get("device", "auto"))
    try:
        runtime_device = resolve_training_device(requested_device)
        if bool(raw_config.get("experiment", {}).get("require_accelerator", False)):
            require_accelerator(
                runtime_device,
                experiment_name=str(raw_config.get("experiment", {}).get("name", "article release")),
            )
    except RuntimeError as error:
        print(f"[release] blocked: {error}", file=sys.stderr)
        return 2

    config_sha256 = snapshot_config(args.config, config_snapshot)
    manifest_path = args.report_root / "release_manifest.json"
    manifest: dict[str, Any] = {
        "kind": "article_release",
        "status": "running",
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "config": {
            "source_path": str(args.config),
            "snapshot_path": str(config_snapshot),
            "sha256": config_sha256,
        },
        "split_seed": args.split_seed,
        "train_seeds": train_seeds,
        "device": {
            "requested": runtime_device.requested,
            "effective": runtime_device.resolved,
        },
        "stages": [{"name": name, "command": command} for name, command in commands],
        "completed_stages": [],
    }
    write_manifest(manifest_path, manifest)

    environment = child_environment(runtime_device.resolved)
    try:
        for stage_name, command in commands:
            manifest["active_stage"] = stage_name
            write_manifest(manifest_path, manifest)
            print(f"[release] {stage_name}")
            print("          " + " ".join(command))
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
            manifest["completed_stages"].append(stage_name)
            write_manifest(manifest_path, manifest)
    except subprocess.CalledProcessError as error:
        manifest["status"] = "failed"
        manifest["failed_stage"] = manifest.get("active_stage")
        manifest["return_code"] = error.returncode
        write_manifest(manifest_path, manifest)
        raise

    manifest.pop("active_stage", None)
    manifest["status"] = "complete"
    write_manifest(manifest_path, manifest)
    print(f"[release] complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
