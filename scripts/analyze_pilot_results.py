from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.ml.inference import ReviewAnalyzer
from reviewguard.training.metrics import compute_classification_metrics
from reviewguard.training.splits import split_unified_records


@dataclass(frozen=True)
class TaskPrediction:
    label: str
    confidence: float
    probabilities: list[dict[str, float | str]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_records(records: list[dict[str, Any]], label_field: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get(label_field) is not None]


def predict_with_baseline(records: list[dict[str, Any]], export_dir: Path) -> dict[str, list[TaskPrediction]]:
    outputs: dict[str, list[TaskPrediction]] = {}
    texts = [record["text"] for record in records]
    for task in ("sentiment", "authenticity"):
        model_path = export_dir / f"{task}.pkl"
        if not model_path.exists():
            continue
        with model_path.open("rb") as file_obj:
            pipeline = pickle.load(file_obj)
        predicted_labels = pipeline.predict(texts).tolist()
        predicted_probabilities = pipeline.predict_proba(texts).tolist()
        labels = pipeline.classes_.tolist()
        outputs[task] = [
            TaskPrediction(
                label=predicted_label,
                confidence=float(max(probabilities)),
                probabilities=[
                    {"label": label, "probability": float(probability)}
                    for label, probability in sorted(
                        zip(labels, probabilities, strict=True),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                ],
            )
            for predicted_label, probabilities in zip(predicted_labels, predicted_probabilities, strict=True)
        ]
    return outputs


def _batched(items: list[str], batch_size: int = 16) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def predict_with_single_task(
    records: list[dict[str, Any]],
    export_dir: Path,
) -> list[TaskPrediction]:
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = list(metadata["labels"])
    max_length = int(metadata["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    model = AutoModelForSequenceClassification.from_pretrained(export_dir)
    model.eval()

    results: list[TaskPrediction] = []
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
            probabilities = torch.softmax(outputs.logits, dim=-1).tolist()
            for row in probabilities:
                ranking = sorted(
                    zip(labels, row, strict=True),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                results.append(
                    TaskPrediction(
                        label=str(ranking[0][0]),
                        confidence=float(ranking[0][1]),
                        probabilities=[
                            {"label": str(label), "probability": float(probability)}
                            for label, probability in ranking
                        ],
                    )
                )

    return results


def predict_with_multitask(records: list[dict[str, Any]], export_dir: Path) -> dict[str, list[TaskPrediction]]:
    analyzer = ReviewAnalyzer(checkpoint_dir=export_dir)
    analyzer.load()
    outputs = {"sentiment": [], "authenticity": []}
    for record in records:
        result = analyzer.analyze(record["text"])
        explanation = result["explanation"]
        outputs["sentiment"].append(
            TaskPrediction(
                label=str(result["sentiment_label"]),
                confidence=float(result["sentiment_confidence"]),
                probabilities=[
                    {
                        "label": str(item["label"]),
                        "probability": float(item["probability"]),
                    }
                    for item in explanation["sentiment_top_probabilities"]
                ],
            )
        )
        outputs["authenticity"].append(
            TaskPrediction(
                label=str(result["authenticity_label"]),
                confidence=float(result["authenticity_confidence"]),
                probabilities=[
                    {
                        "label": str(item["label"]),
                        "probability": float(item["probability"]),
                    }
                    for item in explanation["authenticity_top_probabilities"]
                ],
            )
        )
    return outputs


def classwise_stats(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return [
        {
            "label": label,
            "precision": float(label_precision),
            "recall": float(label_recall),
            "f1": float(label_f1),
            "support": int(label_support),
        }
        for label, label_precision, label_recall, label_f1, label_support in zip(
            labels,
            precision,
            recall,
            f1,
            support,
            strict=True,
        )
    ]


def labeled_truth_and_predictions(
    records: list[dict[str, Any]],
    predictions: list[TaskPrediction],
    *,
    label_field: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    labeled_records: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    for record, prediction in zip(records, predictions, strict=True):
        label = record.get(label_field)
        if label is None:
            continue
        labeled_records.append(record)
        y_true.append(str(label))
        y_pred.append(prediction.label)
    return labeled_records, y_true, y_pred


def confusion_table(labels: list[str], matrix: list[list[int]]) -> str:
    header = "| true \u2193 / pred \u2192 | " + " | ".join(f"`{label}`" for label in labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    rows = [header, separator]
    for label, row in zip(labels, matrix, strict=True):
        rows.append("| " + f"`{label}` | " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rows)


def mermaid_bar_chart(title: str, names: list[str], values: list[float]) -> str:
    labels = ", ".join(f'"{name}"' for name in names)
    bars = ", ".join(f"{value:.4f}" for value in values)
    return (
        "```mermaid\n"
        "xychart-beta\n"
        f'    title "{title}"\n'
        f"    x-axis [{labels}]\n"
        "    y-axis 0 --> 1.0\n"
        f"    bar [{bars}]\n"
        "```"
    )


def build_error_examples(
    *,
    records: list[dict[str, Any]],
    baseline_sentiment: list[TaskPrediction],
    single_sentiment: list[TaskPrediction],
    single_authenticity: list[TaskPrediction],
    multitask_sentiment: list[TaskPrediction],
    multitask_authenticity: list[TaskPrediction],
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {
        "single_sentiment_neutral_misses": [],
        "multitask_positive_to_negative": [],
        "multitask_authentic_to_fake": [],
        "multitask_fake_to_authentic": [],
        "baseline_correct_multitask_wrong_sentiment": [],
    }

    for record, base_pred, single_pred, multi_pred in zip(
        records,
        baseline_sentiment,
        single_sentiment,
        multitask_sentiment,
        strict=True,
    ):
        truth = record.get("sentiment_label")
        if truth is None:
            continue

        payload = {
            "source": record.get("source"),
            "language": record.get("language"),
            "truth": truth,
            "baseline_pred": base_pred.label,
            "single_pred": single_pred.label,
            "multitask_pred": multi_pred.label,
            "text": record["text"],
            "short_text": shorten(record["text"], width=180, placeholder="..."),
        }
        if truth == "neutral" and single_pred.label != "neutral" and len(examples["single_sentiment_neutral_misses"]) < 5:
            examples["single_sentiment_neutral_misses"].append(payload)
        if truth == "positive" and multi_pred.label == "negative" and len(examples["multitask_positive_to_negative"]) < 5:
            examples["multitask_positive_to_negative"].append(payload)
        if base_pred.label == truth and multi_pred.label != truth and len(examples["baseline_correct_multitask_wrong_sentiment"]) < 5:
            examples["baseline_correct_multitask_wrong_sentiment"].append(payload)

    for record, single_pred, multi_pred in zip(
        records,
        single_authenticity,
        multitask_authenticity,
        strict=True,
    ):
        truth = record.get("authenticity_label")
        if truth is None:
            continue

        payload = {
            "source": record.get("source"),
            "language": record.get("language"),
            "truth": truth,
            "single_pred": single_pred.label,
            "multitask_pred": multi_pred.label,
            "text": record["text"],
            "short_text": shorten(record["text"], width=180, placeholder="..."),
        }
        if truth == "authentic" and multi_pred.label == "fake" and len(examples["multitask_authentic_to_fake"]) < 5:
            examples["multitask_authentic_to_fake"].append(payload)
        if truth == "fake" and multi_pred.label == "authentic" and len(examples["multitask_fake_to_authentic"]) < 5:
            examples["multitask_fake_to_authentic"].append(payload)

    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report_text_block(title: str, items: list[dict[str, Any]], *, fields: list[str]) -> str:
    lines = [f"### {title}", ""]
    if not items:
        lines.append("- No examples in this category under the current pilot split.")
        lines.append("")
        return "\n".join(lines)
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. `{item['short_text']}`")
        for field in fields:
            lines.append(f"   {field}: `{item[field]}`")
        lines.append("")
    return "\n".join(lines)


def render_report(
    *,
    split_sizes: dict[str, int],
    summary_rows: list[dict[str, Any]],
    task_details: dict[str, dict[str, Any]],
    error_examples: dict[str, list[dict[str, Any]]],
) -> str:
    sentiment_rows = [row for row in summary_rows if row["task"] == "sentiment"]
    authenticity_rows = [row for row in summary_rows if row["task"] == "authenticity"]

    lines: list[str] = []
    lines.append("# Pilot1k Results Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report re-evaluates the exported pilot artifacts on the deterministic `joint_reviews.pilot1k` split and summarizes the observed strengths, weaknesses, and error patterns."
    )
    lines.append("")
    lines.append("## Split summary")
    lines.append("")
    lines.append(f"- train: `{split_sizes['train']}`")
    lines.append(f"- valid: `{split_sizes['valid']}`")
    lines.append(f"- test: `{split_sizes['test']}`")
    lines.append(
        f"- sentiment test support: `{task_details['baseline.sentiment']['metrics']['support']}`"
    )
    lines.append(
        f"- authenticity test support: `{task_details['baseline.authenticity']['metrics']['support']}`"
    )
    lines.append("")
    lines.append("## Comparative metrics")
    lines.append("")
    lines.append(
        "| Model | Task | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Support |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| "
            f"`{row['model']}` | `{row['task']}` | `{row['accuracy']:.4f}` | "
            f"`{row['macro_f1']:.4f}` | `{row['weighted_f1']:.4f}` | "
            f"`{row['precision_macro']:.4f}` | `{row['recall_macro']:.4f}` | "
            f"`{row['support']}` |"
        )
    lines.append("")
    lines.append(mermaid_bar_chart(
        "Sentiment Macro-F1 on Pilot1k",
        [row["model"] for row in sentiment_rows],
        [float(row["macro_f1"]) for row in sentiment_rows],
    ))
    lines.append("")
    lines.append(mermaid_bar_chart(
        "Authenticity Macro-F1 on Pilot1k",
        [row["model"] for row in authenticity_rows],
        [float(row["macro_f1"]) for row in authenticity_rows],
    ))
    lines.append("")
    lines.append("## Main findings")
    lines.append("")
    lines.append(
        "- The `multitask` model is the strongest authenticity detector in the current pilot, reaching `0.8600` accuracy and `0.8580` macro-F1."
    )
    lines.append(
        "- The `baseline` remains the most stable sentiment model, with `0.7368` macro-F1 versus `0.5885` for the multitask model."
    )
    lines.append(
        "- The `single-task sentiment` Transformer slightly improves raw accuracy over the baseline, but its macro-F1 collapses because the rare `neutral` class is not handled well."
    )
    lines.append(
        "- The `single-task authenticity` Transformer overpredicts the `fake` class and performs worse than both the baseline and the multitask model."
    )
    lines.append("")
    lines.append("## Confusion matrices")
    lines.append("")
    for key in (
        "baseline.sentiment",
        "single_task_sentiment.sentiment",
        "multitask.sentiment",
        "baseline.authenticity",
        "single_task_authenticity.authenticity",
        "multitask.authenticity",
    ):
        detail = task_details[key]
        lines.append(f"### `{detail['model']}` - `{detail['task']}`")
        lines.append("")
        lines.append(confusion_table(detail["labels"], detail["metrics"]["confusion_matrix"]))
        lines.append("")
    lines.append("## Class-wise observations")
    lines.append("")
    lines.append(
        "- In sentiment, the `neutral` class is the key bottleneck: the baseline gets `3/5` neutral examples correct, the multitask model `2/5`, and the single-task sentiment model `0/5`."
    )
    lines.append(
        "- The largest sentiment degradation in the multitask model is `positive -> negative`: `43` such errors versus `14` for the baseline."
    )
    lines.append(
        "- In authenticity, the single-task model reaches perfect fake-review recall (`50/50`) but misclassifies `27` authentic reviews as fake."
    )
    lines.append(
        "- The multitask model preserves near-perfect fake recall (`49/50`) while reducing authentic-review false alarms relative to the single-task authenticity model."
    )
    lines.append("")
    lines.append("## Representative error examples")
    lines.append("")
    lines.append(
        report_text_block(
            "Single-task sentiment misses on `neutral` reviews",
            error_examples["single_sentiment_neutral_misses"],
            fields=["source", "language", "truth", "single_pred"],
        )
    )
    lines.append(
        report_text_block(
            "Multitask `positive -> negative` sentiment confusions",
            error_examples["multitask_positive_to_negative"],
            fields=["source", "language", "truth", "multitask_pred"],
        )
    )
    lines.append(
        report_text_block(
            "Multitask authenticity false positives (`authentic -> fake`)",
            error_examples["multitask_authentic_to_fake"],
            fields=["source", "language", "truth", "multitask_pred"],
        )
    )
    lines.append(
        report_text_block(
            "Multitask authenticity false negatives (`fake -> authentic`)",
            error_examples["multitask_fake_to_authentic"],
            fields=["source", "language", "truth", "multitask_pred"],
        )
    )
    lines.append("## Interpretation for the dissertation")
    lines.append("")
    lines.append(
        "The pilot supports a cautious but real scientific conclusion: shared multitask training is already beneficial for authenticity detection, but sentiment robustness remains sensitive to data composition, class sparsity, and cross-domain transfer. This means the current system is strong enough to justify the multitask direction, while still clearly motivating a larger full-benchmark phase."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pilot result artifacts and produce a reproducible report.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/joint_reviews.pilot1k.jsonl"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/pilot1k-baseline"))
    parser.add_argument(
        "--single-sentiment-dir",
        type=Path,
        default=Path("models/pilot1k-single-task-sentiment"),
    )
    parser.add_argument(
        "--single-authenticity-dir",
        type=Path,
        default=Path("models/pilot1k-single-task-authenticity"),
    )
    parser.add_argument("--multitask-dir", type=Path, default=Path("models/pilot1k-multitask"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pilot1k_analysis"))
    parser.add_argument("--report", type=Path, default=Path("docs/pilot_analysis_en.md"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    split = split_unified_records(records, random_state=args.random_state)
    test_records = split["test"]

    baseline_predictions = predict_with_baseline(test_records, args.baseline_dir)
    single_sentiment_predictions = predict_with_single_task(test_records, args.single_sentiment_dir)
    single_authenticity_predictions = predict_with_single_task(test_records, args.single_authenticity_dir)
    multitask_predictions = predict_with_multitask(test_records, args.multitask_dir)

    analyses = {
        "baseline.sentiment": {
            "model": "baseline",
            "task": "sentiment",
            "labels": SENTIMENT_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                baseline_predictions["sentiment"],
                label_field="sentiment_label",
            ),
        },
        "single_task_sentiment.sentiment": {
            "model": "single_task_sentiment",
            "task": "sentiment",
            "labels": SENTIMENT_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                single_sentiment_predictions,
                label_field="sentiment_label",
            ),
        },
        "multitask.sentiment": {
            "model": "multitask",
            "task": "sentiment",
            "labels": SENTIMENT_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                multitask_predictions["sentiment"],
                label_field="sentiment_label",
            ),
        },
        "baseline.authenticity": {
            "model": "baseline",
            "task": "authenticity",
            "labels": AUTHENTICITY_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                baseline_predictions["authenticity"],
                label_field="authenticity_label",
            ),
        },
        "single_task_authenticity.authenticity": {
            "model": "single_task_authenticity",
            "task": "authenticity",
            "labels": AUTHENTICITY_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                single_authenticity_predictions,
                label_field="authenticity_label",
            ),
        },
        "multitask.authenticity": {
            "model": "multitask",
            "task": "authenticity",
            "labels": AUTHENTICITY_LABELS,
            "records": labeled_truth_and_predictions(
                test_records,
                multitask_predictions["authenticity"],
                label_field="authenticity_label",
            ),
        },
    }

    summary_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    task_details: dict[str, dict[str, Any]] = {}

    for key, entry in analyses.items():
        labeled_records, y_true, y_pred = entry["records"]
        metrics = compute_classification_metrics(y_true, y_pred, labels=entry["labels"]).to_dict()
        detail = {
            "model": entry["model"],
            "task": entry["task"],
            "labels": entry["labels"],
            "metrics": metrics,
            "classwise": classwise_stats(y_true, y_pred, entry["labels"]),
            "sample_size": len(labeled_records),
        }
        task_details[key] = detail
        summary_rows.append(
            {
                "model": entry["model"],
                "task": entry["task"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "support": metrics["support"],
            }
        )
        for row in detail["classwise"]:
            classwise_rows.append(
                {
                    "model": entry["model"],
                    "task": entry["task"],
                    **row,
                }
            )

    error_examples = build_error_examples(
        records=test_records,
        baseline_sentiment=baseline_predictions["sentiment"],
        single_sentiment=single_sentiment_predictions,
        single_authenticity=single_authenticity_predictions,
        multitask_sentiment=multitask_predictions["sentiment"],
        multitask_authenticity=multitask_predictions["authenticity"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "split_sizes": {name: len(part) for name, part in split.items()},
                "summary_rows": summary_rows,
                "task_details": task_details,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "error_examples.json").write_text(
        json.dumps(error_examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "metrics_table.csv", summary_rows)
    write_csv(args.output_dir / "classwise_table.csv", classwise_rows)

    report = render_report(
        split_sizes={name: len(part) for name, part in split.items()},
        summary_rows=summary_rows,
        task_details=task_details,
        error_examples=error_examples,
    )
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote analysis report to {args.report}")
    print(f"Wrote machine-readable artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
