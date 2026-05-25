from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


LABEL_FIELDS = {
    "sentiment": "sentiment_label",
    "authenticity": "authenticity_label",
}


@dataclass(frozen=True)
class TaskMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision_macro: float
    recall_macro: float
    support: int
    per_label_support: dict[str, int]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str],
) -> TaskMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one labeled example is required to compute metrics.")

    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_label_support = {label: int(label_support) for label, label_support in zip(labels, support)}

    precision_macro = float(sum(precision) / len(labels))
    recall_macro = float(sum(recall) / len(labels))

    return TaskMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        support=len(y_true),
        per_label_support=per_label_support,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def _task_truth_and_predictions(
    records: list[dict[str, Any]],
    predictions: list[str | None],
    *,
    task: str,
) -> tuple[list[str], list[str]]:
    label_field = LABEL_FIELDS[task]
    if len(records) != len(predictions):
        raise ValueError(f"{task} predictions must align one-to-one with records.")

    y_true: list[str] = []
    y_pred: list[str] = []
    for record, prediction in zip(records, predictions):
        label = record.get(label_field)
        if label is None:
            continue
        if prediction is None:
            raise ValueError(f"{task} prediction is missing for a labeled example.")
        y_true.append(label)
        y_pred.append(prediction)

    return y_true, y_pred


def compute_multitask_metrics(
    records: list[dict[str, Any]],
    *,
    sentiment_predictions: list[str | None] | None = None,
    authenticity_predictions: list[str | None] | None = None,
    sentiment_labels: list[str],
    authenticity_labels: list[str],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}

    if sentiment_predictions is not None:
        y_true, y_pred = _task_truth_and_predictions(
            records,
            sentiment_predictions,
            task="sentiment",
        )
        if y_true:
            metrics["sentiment"] = compute_classification_metrics(
                y_true,
                y_pred,
                labels=sentiment_labels,
            ).to_dict()

    if authenticity_predictions is not None:
        y_true, y_pred = _task_truth_and_predictions(
            records,
            authenticity_predictions,
            task="authenticity",
        )
        if y_true:
            metrics["authenticity"] = compute_classification_metrics(
                y_true,
                y_pred,
                labels=authenticity_labels,
            ).to_dict()

    return metrics
