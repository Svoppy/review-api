from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Literal

import numpy as np


MetricName = Literal["accuracy", "macro_f1"]

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one value is required to compute a confidence interval.")
    if confidence != 0.95:
        raise ValueError("Only 95% confidence intervals are currently supported.")

    sample_mean = mean(values)
    if len(values) == 1:
        return {
            "mean": sample_mean,
            "std": 0.0,
            "ci_low": sample_mean,
            "ci_high": sample_mean,
            "n": 1,
        }

    sample_std = stdev(values)
    df = len(values) - 1
    t_critical = _T_CRITICAL_95.get(df, 1.96)
    margin = t_critical * (sample_std / math.sqrt(len(values)))
    ci_low = sample_mean - margin
    ci_high = sample_mean + margin
    if lower_bound is not None:
        ci_low = max(lower_bound, ci_low)
    if upper_bound is not None:
        ci_high = min(upper_bound, ci_high)
    return {
        "mean": sample_mean,
        "std": sample_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(values),
    }


def metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: MetricName,
    num_labels: int,
) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 1:
        raise ValueError("Only flat label arrays are supported.")

    if metric == "accuracy":
        return float(np.mean(y_true == y_pred))
    if metric != "macro_f1":
        raise ValueError(f"Unsupported metric: {metric}")

    f1_scores: list[float] = []
    for label_id in range(num_labels):
        true_positive = int(np.sum((y_true == label_id) & (y_pred == label_id)))
        false_positive = int(np.sum((y_true != label_id) & (y_pred == label_id)))
        false_negative = int(np.sum((y_true == label_id) & (y_pred != label_id)))
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(sum(f1_scores) / num_labels)


def paired_seed_delta(
    y_true: np.ndarray,
    model_a_predictions: list[np.ndarray],
    model_b_predictions: list[np.ndarray],
    *,
    metric: MetricName,
    num_labels: int,
) -> tuple[list[float], float]:
    if len(model_a_predictions) != len(model_b_predictions):
        raise ValueError("Model A and Model B must have the same number of seed-specific predictions.")

    deltas: list[float] = []
    for y_pred_a, y_pred_b in zip(model_a_predictions, model_b_predictions, strict=True):
        metric_a = metric_value(y_true, y_pred_a, metric=metric, num_labels=num_labels)
        metric_b = metric_value(y_true, y_pred_b, metric=metric, num_labels=num_labels)
        deltas.append(metric_a - metric_b)
    return deltas, float(mean(deltas))


def paired_bootstrap_confidence_interval(
    y_true: np.ndarray,
    model_a_predictions: list[np.ndarray],
    model_b_predictions: list[np.ndarray],
    *,
    metric: MetricName,
    num_labels: int,
    iterations: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    rng = np.random.default_rng(random_state)
    sample_size = int(y_true.shape[0])
    deltas = np.empty(iterations, dtype=float)

    for index in range(iterations):
        sample_indices = rng.integers(0, sample_size, size=sample_size)
        sampled_truth = y_true[sample_indices]
        seed_deltas: list[float] = []
        for y_pred_a, y_pred_b in zip(model_a_predictions, model_b_predictions, strict=True):
            sampled_pred_a = y_pred_a[sample_indices]
            sampled_pred_b = y_pred_b[sample_indices]
            seed_deltas.append(
                metric_value(
                    sampled_truth,
                    sampled_pred_a,
                    metric=metric,
                    num_labels=num_labels,
                )
                - metric_value(
                    sampled_truth,
                    sampled_pred_b,
                    metric=metric,
                    num_labels=num_labels,
                )
            )
        deltas[index] = float(mean(seed_deltas))

    return {
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def approximate_randomization_test(
    y_true: np.ndarray,
    model_a_predictions: list[np.ndarray],
    model_b_predictions: list[np.ndarray],
    *,
    metric: MetricName,
    num_labels: int,
    iterations: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    seed_deltas, observed_delta = paired_seed_delta(
        y_true,
        model_a_predictions,
        model_b_predictions,
        metric=metric,
        num_labels=num_labels,
    )
    _ = seed_deltas

    rng = np.random.default_rng(random_state)
    sample_size = int(y_true.shape[0])
    exceedances = 0

    for _ in range(iterations):
        permuted_seed_deltas: list[float] = []
        for y_pred_a, y_pred_b in zip(model_a_predictions, model_b_predictions, strict=True):
            swap_mask = rng.integers(0, 2, size=sample_size, dtype=np.int8).astype(bool)
            permuted_a = np.where(swap_mask, y_pred_b, y_pred_a)
            permuted_b = np.where(swap_mask, y_pred_a, y_pred_b)
            permuted_seed_deltas.append(
                metric_value(
                    y_true,
                    permuted_a,
                    metric=metric,
                    num_labels=num_labels,
                )
                - metric_value(
                    y_true,
                    permuted_b,
                    metric=metric,
                    num_labels=num_labels,
                )
            )

        permuted_delta = float(mean(permuted_seed_deltas))
        if abs(permuted_delta) >= abs(observed_delta):
            exceedances += 1

    p_value = (exceedances + 1) / (iterations + 1)
    return {
        "observed_delta": observed_delta,
        "p_value": p_value,
    }
