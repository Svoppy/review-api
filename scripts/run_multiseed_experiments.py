from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_MODELS = (
    "baseline",
    "single-task-sentiment",
    "single-task-authenticity",
    "multitask",
)
DEFAULT_TRAIN_SEEDS = (11, 21, 42)
NUMERIC_METRICS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
    "support",
)


@dataclass(frozen=True)
class ExperimentSpec:
    model_id: str
    subcommand: str
    task: str | None = None
    uses_config: bool = False


EXPERIMENT_SPECS = {
    "baseline": ExperimentSpec(model_id="baseline", subcommand="baseline"),
    "single-task-sentiment": ExperimentSpec(
        model_id="single-task-sentiment",
        subcommand="single-task",
        task="sentiment",
        uses_config=True,
    ),
    "single-task-authenticity": ExperimentSpec(
        model_id="single-task-authenticity",
        subcommand="single-task",
        task="authenticity",
        uses_config=True,
    ),
    "multitask": ExperimentSpec(
        model_id="multitask",
        subcommand="multitask",
        uses_config=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline, single-task, and multitask training across multiple train seeds "
            "while keeping the split seed fixed."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/joint_reviews.pilot1k.jsonl"),
        help="Unified JSONL corpus to train on.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model.pilot.yaml"),
        help="YAML config for Transformer-based runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/multiseed/pilot1k"),
        help="Root directory where per-seed model exports will be stored.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/multiseed/pilot1k"),
        help="Directory for aggregated multi-seed summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EXPERIMENT_SPECS.keys()),
        default=list(DEFAULT_MODELS),
        help="Subset of experiments to run.",
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRAIN_SEEDS),
        help="Training seeds to sweep over while holding the split fixed.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed random seed for the train/validation/test split.",
    )
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child training runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed runs when train_report.json already exists.",
    )
    return parser.parse_args()


def seed_tag(seed: int) -> str:
    return f"seed-{seed}"


def export_dir_for(output_root: Path, model_id: str, train_seed: int) -> Path:
    return output_root / model_id / seed_tag(train_seed)


def report_path_for(output_root: Path, model_id: str, train_seed: int) -> Path:
    return export_dir_for(output_root, model_id, train_seed) / "train_report.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_child_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join((src_path, existing))
    return env


def build_command(
    *,
    python_executable: str,
    input_path: Path,
    config_path: Path,
    export_dir: Path,
    spec: ExperimentSpec,
    split_seed: int,
    train_seed: int,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "reviewguard.training",
        spec.subcommand,
        "--input",
        str(input_path),
        "--export-dir",
        str(export_dir),
        "--train-size",
        str(train_size),
        "--valid-size",
        str(valid_size),
        "--test-size",
        str(test_size),
        "--split-random-state",
        str(split_seed),
        "--train-random-state",
        str(train_seed),
    ]
    if spec.task is not None:
        command.extend(["--task", spec.task])
    if spec.uses_config:
        command.extend(["--config", str(config_path)])
    return command


def run_one_experiment(
    *,
    repo_root: Path,
    python_executable: str,
    input_path: Path,
    config_path: Path,
    output_root: Path,
    spec: ExperimentSpec,
    split_seed: int,
    train_seed: int,
    train_size: float,
    valid_size: float,
    test_size: float,
    skip_existing: bool,
) -> dict[str, Any]:
    export_dir = export_dir_for(output_root, spec.model_id, train_seed)
    report_path = report_path_for(output_root, spec.model_id, train_seed)

    if skip_existing and report_path.exists():
        print(f"[reuse] {spec.model_id} {seed_tag(train_seed)}")
        report = read_json(report_path)
    else:
        export_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            python_executable=python_executable,
            input_path=input_path,
            config_path=config_path,
            export_dir=export_dir,
            spec=spec,
            split_seed=split_seed,
            train_seed=train_seed,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
        )
        print(f"[run] {spec.model_id} {seed_tag(train_seed)}")
        print("      " + " ".join(command))
        subprocess.run(
            command,
            check=True,
            cwd=repo_root,
            env=make_child_env(repo_root),
        )
        report = read_json(report_path)

    return {
        "model_id": spec.model_id,
        "train_seed": train_seed,
        "split_seed": split_seed,
        "export_dir": str(export_dir),
        "report_path": str(report_path),
        "report": report,
    }


def _metric_aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    metric_mean = mean(values)
    metric_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": metric_mean,
        "std": 0.0 if math.isclose(metric_std, 0.0) else metric_std,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values": values,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(str(run["model_id"]), []).append(run)

    aggregates: dict[str, Any] = {}
    for model_id, model_runs in sorted(by_model.items()):
        model_summary: dict[str, Any] = {
            "train_seeds": [int(run["train_seed"]) for run in model_runs],
            "split_seed": int(model_runs[0]["split_seed"]),
            "validation_metrics": {},
            "test_metrics": {},
        }
        for split_name in ("validation_metrics", "test_metrics"):
            split_summary: dict[str, Any] = {}
            tasks = sorted(
                {
                    task
                    for run in model_runs
                    for task in (run["report"].get(split_name) or {}).keys()
                }
            )
            for task in tasks:
                task_summary: dict[str, Any] = {}
                for metric_name in NUMERIC_METRICS:
                    values = [
                        float(task_metrics[metric_name])
                        for run in model_runs
                        if (task_metrics := (run["report"].get(split_name) or {}).get(task))
                        and metric_name in task_metrics
                    ]
                    if values:
                        task_summary[metric_name] = _metric_aggregate(values)
                if task_summary:
                    split_summary[task] = task_summary
            model_summary[split_name] = split_summary
        aggregates[model_id] = model_summary

    return aggregates


def aggregates_to_csv_rows(aggregates: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model_summary in sorted(aggregates.items()):
        train_seeds = list(model_summary["train_seeds"])
        for split_name in ("validation_metrics", "test_metrics"):
            for task, task_summary in sorted(model_summary[split_name].items()):
                for metric_name, metric_summary in sorted(task_summary.items()):
                    rows.append(
                        {
                            "model": model_id,
                            "split": split_name.removesuffix("_metrics"),
                            "task": task,
                            "metric": metric_name,
                            "n": metric_summary["n"],
                            "mean": round(float(metric_summary["mean"]), 6),
                            "std": round(float(metric_summary["std"]), 6),
                            "min": round(float(metric_summary["min"]), 6),
                            "max": round(float(metric_summary["max"]), 6),
                            "train_seeds": ",".join(str(seed) for seed in train_seeds),
                            "values": json.dumps(metric_summary["values"]),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = str(row[header]).replace('"', '""')
            if any(char in value for char in (",", '"', "\n")):
                value = f'"{value}"'
            values.append(value)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_markdown_summary(
    *,
    input_path: Path,
    split_seed: int,
    train_seeds: list[int],
    aggregates: dict[str, Any],
) -> str:
    lines = [
        "# Multi-Seed Experiment Summary",
        "",
        f"- Input: `{input_path}`",
        f"- Fixed split seed: `{split_seed}`",
        f"- Training seeds: `{', '.join(str(seed) for seed in train_seeds)}`",
        "",
        "## Test Macro-F1",
        "",
        "| Model | Task | Macro-F1 | Accuracy |",
        "|---|---|---:|---:|",
    ]
    for model_id, model_summary in sorted(aggregates.items()):
        for task, task_summary in sorted(model_summary["test_metrics"].items()):
            macro_f1 = task_summary.get("macro_f1")
            accuracy = task_summary.get("accuracy")
            if not macro_f1 or not accuracy:
                continue
            lines.append(
                "| "
                + f"`{model_id}` | `{task}` | "
                + f"{macro_f1['mean']:.4f} +/- {macro_f1['std']:.4f} | "
                + f"{accuracy['mean']:.4f} +/- {accuracy['std']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root
    report_dir = args.report_dir
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for model_id in args.models:
        spec = EXPERIMENT_SPECS[model_id]
        for train_seed in args.train_seeds:
            runs.append(
                run_one_experiment(
                    repo_root=repo_root,
                    python_executable=args.python,
                    input_path=args.input,
                    config_path=args.config,
                    output_root=output_root,
                    spec=spec,
                    split_seed=args.split_seed,
                    train_seed=train_seed,
                    train_size=args.train_size,
                    valid_size=args.valid_size,
                    test_size=args.test_size,
                    skip_existing=args.skip_existing,
                )
            )

    aggregates = aggregate_runs(runs)
    summary = {
        "input_path": str(args.input),
        "config_path": str(args.config),
        "split_seed": args.split_seed,
        "train_seeds": list(args.train_seeds),
        "models": list(args.models),
        "runs": [
            {
                "model_id": run["model_id"],
                "train_seed": run["train_seed"],
                "split_seed": run["split_seed"],
                "export_dir": run["export_dir"],
                "report_path": run["report_path"],
            }
            for run in runs
        ],
        "aggregates": aggregates,
    }

    summary_path = report_dir / "summary.json"
    csv_path = report_dir / "metrics_table.csv"
    markdown_path = report_dir / "summary.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, aggregates_to_csv_rows(aggregates))
    markdown_path.write_text(
        build_markdown_summary(
            input_path=args.input,
            split_seed=args.split_seed,
            train_seeds=list(args.train_seeds),
            aggregates=aggregates,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
