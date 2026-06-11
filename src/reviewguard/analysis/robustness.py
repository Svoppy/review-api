from __future__ import annotations

from statistics import mean
from typing import Any

from reviewguard.training.metrics import compute_classification_metrics, label_field_for_task


def _slice_value(record: dict[str, Any], *, slice_field: str) -> str:
    value = record.get(slice_field)
    if value in (None, ""):
        return "unknown"
    return str(value)


def build_slice_metrics(
    records: list[dict[str, Any]],
    *,
    predictions: list[str | None] | None,
    task: str,
    labels: list[str],
    slice_field: str,
    min_support: int = 2,
) -> dict[str, Any]:
    if predictions is None:
        return {
            "slice_field": slice_field,
            "task": task,
            "slices": [],
            "macro_f1_mean_across_slices": None,
            "worst_slice_macro_f1": None,
            "best_slice_macro_f1": None,
            "robustness_gap": None,
            "evaluated_slices": 0,
            "min_support": min_support,
        }

    label_field = label_field_for_task(task)
    grouped_truth: dict[str, list[str]] = {}
    grouped_pred: dict[str, list[str]] = {}

    for record, prediction in zip(records, predictions, strict=True):
        label = record.get(label_field)
        if label is None:
            continue
        if prediction is None:
            raise ValueError(f"{task} prediction is missing for a labeled example.")
        group = _slice_value(record, slice_field=slice_field)
        grouped_truth.setdefault(group, []).append(str(label))
        grouped_pred.setdefault(group, []).append(str(prediction))

    rows: list[dict[str, Any]] = []
    macro_f1_values: list[float] = []
    for group in sorted(grouped_truth):
        y_true = grouped_truth[group]
        if len(y_true) < min_support:
            continue
        metrics = compute_classification_metrics(
            y_true,
            grouped_pred[group],
            labels=labels,
        ).to_dict()
        metrics["slice_value"] = group
        rows.append(metrics)
        macro_f1_values.append(float(metrics["macro_f1"]))

    if not macro_f1_values:
        mean_macro_f1 = None
        worst_slice = None
        best_slice = None
        robustness_gap = None
    else:
        mean_macro_f1 = float(mean(macro_f1_values))
        worst_slice = min(macro_f1_values)
        best_slice = max(macro_f1_values)
        robustness_gap = float(best_slice - worst_slice)

    return {
        "slice_field": slice_field,
        "task": task,
        "slices": rows,
        "macro_f1_mean_across_slices": mean_macro_f1,
        "worst_slice_macro_f1": worst_slice,
        "best_slice_macro_f1": best_slice,
        "robustness_gap": robustness_gap,
        "evaluated_slices": len(rows),
        "min_support": min_support,
    }


def build_multitask_robustness_report(
    records: list[dict[str, Any]],
    *,
    sentiment_predictions: list[str | None] | None,
    authenticity_predictions: list[str | None] | None,
    sentiment_labels: list[str],
    authenticity_labels: list[str],
    slice_fields: tuple[str, ...] = ("source", "domain", "language"),
    min_support: int = 2,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for task, predictions, labels in (
        ("sentiment", sentiment_predictions, sentiment_labels),
        ("authenticity", authenticity_predictions, authenticity_labels),
    ):
        report[task] = {
            slice_field: build_slice_metrics(
                records,
                predictions=predictions,
                task=task,
                labels=labels,
                slice_field=slice_field,
                min_support=min_support,
            )
            for slice_field in slice_fields
        }
    return report
