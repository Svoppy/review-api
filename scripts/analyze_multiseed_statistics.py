from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviewguard.analysis.statistics import (
    approximate_randomization_test,
    mean_confidence_interval,
    metric_value,
    paired_bootstrap_confidence_interval,
    paired_seed_delta,
)
from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.ml.inference import ReviewAnalyzer
from reviewguard.training.splits import split_unified_records


DEFAULT_COMPARISONS = (
    "authenticity:multitask:single-task-authenticity",
    "authenticity:multitask:baseline",
    "sentiment:multitask:single-task-sentiment",
    "sentiment:multitask:baseline",
)
DEFAULT_MODELS = (
    "baseline",
    "single-task-sentiment",
    "single-task-authenticity",
    "multitask",
)
MODEL_TO_DIR = {
    "baseline": "baseline",
    "single-task-sentiment": "single-task-sentiment",
    "single-task-authenticity": "single-task-authenticity",
    "multitask": "multitask",
}
TASK_LABELS = {
    "sentiment": SENTIMENT_LABELS,
    "authenticity": AUTHENTICITY_LABELS,
}
TASK_LABEL_FIELDS = {
    "sentiment": "sentiment_label",
    "authenticity": "authenticity_label",
}


@dataclass(frozen=True)
class ComparisonSpec:
    task: str
    model_a: str
    model_b: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute confidence intervals and paired significance-oriented comparisons "
            "for multi-seed ReviewGuard experiments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/joint_reviews.pilot1k.jsonl"),
    )
    parser.add_argument(
        "--multiseed-root",
        type=Path,
        default=Path("models/multiseed/pilot1k"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/multiseed/pilot1k/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/multiseed/pilot1k_statistics"),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[11, 21, 42],
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--randomization-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=list(DEFAULT_COMPARISONS),
        help="Triples formatted as task:model_a:model_b.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_comparison_spec(raw_value: str) -> ComparisonSpec:
    parts = raw_value.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid comparison spec '{raw_value}'. Expected format task:model_a:model_b."
        )
    task, model_a, model_b = parts
    if task not in TASK_LABELS:
        raise ValueError(f"Unsupported task in comparison spec: {task}")
    if model_a not in MODEL_TO_DIR or model_b not in MODEL_TO_DIR:
        raise ValueError(f"Unsupported model id in comparison spec: {raw_value}")
    return ComparisonSpec(task=task, model_a=model_a, model_b=model_b)


def _batched(items: list[str], batch_size: int = 16) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def predict_baseline(records: list[dict[str, Any]], export_dir: Path, task: str) -> list[str]:
    with (export_dir / f"{task}.pkl").open("rb") as file_obj:
        pipeline = pickle.load(file_obj)
    return [str(label) for label in pipeline.predict([record["text"] for record in records]).tolist()]


def predict_single_task(records: list[dict[str, Any]], export_dir: Path) -> list[str]:
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = list(metadata["labels"])
    max_length = int(metadata["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    model = AutoModelForSequenceClassification.from_pretrained(export_dir)
    model.eval()

    predictions: list[str] = []
    with torch.inference_mode():
        for batch_texts in _batched([record["text"] for record in records]):
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            predicted_ids = outputs.logits.argmax(dim=-1).tolist()
            predictions.extend(str(labels[index]) for index in predicted_ids)
    return predictions


def predict_multitask(records: list[dict[str, Any]], export_dir: Path, task: str) -> list[str]:
    analyzer = ReviewAnalyzer(checkpoint_dir=export_dir)
    analyzer.load()
    label_key = f"{task}_label"
    return [str(analyzer.analyze(record["text"])[label_key]) for record in records]


def model_predictions_for_task(
    records: list[dict[str, Any]],
    *,
    model_id: str,
    export_dir: Path,
    task: str,
) -> list[str]:
    if model_id == "baseline":
        return predict_baseline(records, export_dir, task)
    if model_id.startswith("single-task"):
        return predict_single_task(records, export_dir)
    if model_id == "multitask":
        return predict_multitask(records, export_dir, task)
    raise ValueError(f"Unsupported model id: {model_id}")


def label_records(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    label_field = TASK_LABEL_FIELDS[task]
    return [record for record in records if record.get(label_field) is not None]


def encode_labels(records: list[dict[str, Any]], task: str) -> tuple[np.ndarray, dict[str, int]]:
    labels = list(TASK_LABELS[task])
    label_to_id = {label: index for index, label in enumerate(labels)}
    label_field = TASK_LABEL_FIELDS[task]
    y_true = np.array([label_to_id[str(record[label_field])] for record in records], dtype=np.int64)
    return y_true, label_to_id


def encode_predictions(predictions: list[str], label_to_id: dict[str, int]) -> np.ndarray:
    return np.array([label_to_id[label] for label in predictions], dtype=np.int64)


def model_interval_rows(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for model_id, model_summary in sorted(summary["aggregates"].items()):
        model_payload: dict[str, Any] = {"test_metrics": {}, "validation_metrics": {}}
        for split_name in ("test_metrics", "validation_metrics"):
            split_payload: dict[str, Any] = {}
            for task, task_summary in sorted(model_summary[split_name].items()):
                task_payload: dict[str, Any] = {}
                for metric_name, metric_summary in sorted(task_summary.items()):
                    interval = mean_confidence_interval(
                        [float(value) for value in metric_summary["values"]],
                        lower_bound=0.0 if metric_name != "support" else None,
                        upper_bound=1.0 if metric_name != "support" else None,
                    )
                    task_payload[metric_name] = interval
                    rows.append(
                        {
                            "model": model_id,
                            "split": split_name.removesuffix("_metrics"),
                            "task": task,
                            "metric": metric_name,
                            "mean": round(float(interval["mean"]), 6),
                            "std": round(float(interval["std"]), 6),
                            "ci_low": round(float(interval["ci_low"]), 6),
                            "ci_high": round(float(interval["ci_high"]), 6),
                            "n": int(interval["n"]),
                        }
                    )
                split_payload[task] = task_payload
            model_payload[split_name] = split_payload
        payload[model_id] = model_payload
    return rows, payload


def comparison_rows(
    *,
    test_records_by_task: dict[str, list[dict[str, Any]]],
    predictions_by_model: dict[str, dict[int, dict[str, list[str]]]],
    train_seeds: list[int],
    comparisons: list[ComparisonSpec],
    bootstrap_samples: int,
    randomization_samples: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payload: list[dict[str, Any]] = []

    for comparison in comparisons:
        task_records = test_records_by_task[comparison.task]
        y_true, label_to_id = encode_labels(task_records, comparison.task)
        num_labels = len(label_to_id)

        model_a_predictions = [
            encode_predictions(
                predictions_by_model[comparison.model_a][seed][comparison.task],
                label_to_id,
            )
            for seed in train_seeds
        ]
        model_b_predictions = [
            encode_predictions(
                predictions_by_model[comparison.model_b][seed][comparison.task],
                label_to_id,
            )
            for seed in train_seeds
        ]

        comparison_payload: dict[str, Any] = {
            "task": comparison.task,
            "model_a": comparison.model_a,
            "model_b": comparison.model_b,
            "n_examples": int(y_true.shape[0]),
            "train_seeds": train_seeds,
            "metrics": {},
        }
        for metric_name in ("accuracy", "macro_f1"):
            per_seed_deltas, observed_delta = paired_seed_delta(
                y_true,
                model_a_predictions,
                model_b_predictions,
                metric=metric_name,
                num_labels=num_labels,
            )
            bootstrap_interval = paired_bootstrap_confidence_interval(
                y_true,
                model_a_predictions,
                model_b_predictions,
                metric=metric_name,
                num_labels=num_labels,
                iterations=bootstrap_samples,
                random_state=random_state,
            )
            randomization = approximate_randomization_test(
                y_true,
                model_a_predictions,
                model_b_predictions,
                metric=metric_name,
                num_labels=num_labels,
                iterations=randomization_samples,
                random_state=random_state,
            )
            comparison_payload["metrics"][metric_name] = {
                "delta_per_seed": per_seed_deltas,
                "observed_delta": observed_delta,
                "bootstrap_ci95": bootstrap_interval,
                "approx_randomization_p_value": randomization["p_value"],
            }
            rows.append(
                {
                    "task": comparison.task,
                    "metric": metric_name,
                    "model_a": comparison.model_a,
                    "model_b": comparison.model_b,
                    "observed_delta": round(float(observed_delta), 6),
                    "ci_low": round(float(bootstrap_interval["ci_low"]), 6),
                    "ci_high": round(float(bootstrap_interval["ci_high"]), 6),
                    "p_value": round(float(randomization["p_value"]), 6),
                    "n_examples": int(y_true.shape[0]),
                }
            )

        payload.append(comparison_payload)

    return rows, payload


def build_report(
    *,
    input_path: Path,
    split_seed: int,
    train_seeds: list[int],
    interval_rows: list[dict[str, Any]],
    comparison_rows_data: list[dict[str, Any]],
) -> str:
    lines = [
        "# Multi-Seed Statistical Analysis",
        "",
        f"- Input: `{input_path}`",
        f"- Fixed split seed: `{split_seed}`",
        f"- Training seeds: `{', '.join(str(seed) for seed in train_seeds)}`",
        "",
        "## Confidence intervals across train seeds",
        "",
        "| Model | Split | Task | Metric | Mean | Std | 95% CI |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in interval_rows:
        if row["split"] != "test" or row["metric"] not in {"accuracy", "macro_f1"}:
            continue
        lines.append(
            "| "
            f"`{row['model']}` | `{row['split']}` | `{row['task']}` | `{row['metric']}` | "
            f"{row['mean']:.4f} | {row['std']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired model comparisons on the fixed test split",
            "",
            "| Task | Metric | A | B | Mean Delta | 95% Bootstrap CI | Approx. Randomization p |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison_rows_data:
        lines.append(
            "| "
            f"`{row['task']}` | `{row['metric']}` | `{row['model_a']}` | `{row['model_b']}` | "
            f"{row['observed_delta']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Confidence intervals for model-level metrics are Student-t 95% intervals over train seeds.",
            "- Pairwise deltas use paired bootstrap resampling over the fixed test examples, averaged across train seeds.",
            "- p-values come from an approximate randomization test on the same fixed test split.",
            "- With only three train seeds, this layer is still preliminary; it is meant to prevent overclaiming, not to overstate certainty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    records = read_jsonl(args.input)
    split = split_unified_records(records, random_state=args.split_seed)
    test_records = split["test"]

    test_records_by_task = {
        task: label_records(test_records, task) for task in ("sentiment", "authenticity")
    }

    predictions_by_model: dict[str, dict[int, dict[str, list[str]]]] = {
        model_id: {} for model_id in DEFAULT_MODELS
    }
    for model_id in DEFAULT_MODELS:
        model_dir_name = MODEL_TO_DIR[model_id]
        for seed in args.train_seeds:
            export_dir = args.multiseed_root / model_dir_name / f"seed-{seed}"
            task_predictions: dict[str, list[str]] = {}
            if model_id in {"baseline", "multitask"}:
                for task in ("sentiment", "authenticity"):
                    task_predictions[task] = model_predictions_for_task(
                        test_records_by_task[task],
                        model_id=model_id,
                        export_dir=export_dir,
                        task=task,
                    )
            elif model_id == "single-task-sentiment":
                task_predictions["sentiment"] = model_predictions_for_task(
                    test_records_by_task["sentiment"],
                    model_id=model_id,
                    export_dir=export_dir,
                    task="sentiment",
                )
            elif model_id == "single-task-authenticity":
                task_predictions["authenticity"] = model_predictions_for_task(
                    test_records_by_task["authenticity"],
                    model_id=model_id,
                    export_dir=export_dir,
                    task="authenticity",
                )
            predictions_by_model[model_id][seed] = task_predictions

    interval_rows, interval_payload = model_interval_rows(summary)
    comparisons = [parse_comparison_spec(raw_value) for raw_value in args.comparisons]
    comparison_rows_data, comparisons_payload = comparison_rows(
        test_records_by_task=test_records_by_task,
        predictions_by_model=predictions_by_model,
        train_seeds=list(args.train_seeds),
        comparisons=comparisons,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        random_state=args.random_state,
    )

    summary_payload = {
        "input_path": str(args.input),
        "split_seed": args.split_seed,
        "train_seeds": list(args.train_seeds),
        "bootstrap_samples": args.bootstrap_samples,
        "randomization_samples": args.randomization_samples,
        "model_intervals": interval_payload,
        "pairwise_comparisons": comparisons_payload,
    }

    (args.output_dir / "statistics_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "model_intervals.csv", interval_rows)
    write_csv(args.output_dir / "pairwise_comparisons.csv", comparison_rows_data)
    (args.output_dir / "statistics_report.md").write_text(
        build_report(
            input_path=args.input,
            split_seed=args.split_seed,
            train_seeds=list(args.train_seeds),
            interval_rows=interval_rows,
            comparison_rows_data=comparison_rows_data,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote statistical summary to {args.output_dir / 'statistics_summary.json'}")
    print(f"Wrote model intervals to {args.output_dir / 'model_intervals.csv'}")
    print(f"Wrote pairwise comparisons to {args.output_dir / 'pairwise_comparisons.csv'}")
    print(f"Wrote report to {args.output_dir / 'statistics_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
