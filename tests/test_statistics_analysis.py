from __future__ import annotations

import numpy as np

from reviewguard.analysis.statistics import (
    approximate_randomization_test,
    mean_confidence_interval,
    metric_value,
    paired_bootstrap_confidence_interval,
    paired_seed_delta,
)


def test_mean_confidence_interval_returns_degenerate_interval_for_single_value() -> None:
    interval = mean_confidence_interval([0.8])

    assert interval["mean"] == 0.8
    assert interval["std"] == 0.0
    assert interval["ci_low"] == 0.8
    assert interval["ci_high"] == 0.8


def test_mean_confidence_interval_can_clip_to_metric_bounds() -> None:
    interval = mean_confidence_interval([0.86, 0.93, 0.96], lower_bound=0.0, upper_bound=1.0)

    assert 0.0 <= interval["ci_low"] <= interval["ci_high"] <= 1.0


def test_metric_value_supports_accuracy_and_macro_f1() -> None:
    y_true = np.array([0, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 2], dtype=np.int64)

    assert metric_value(y_true, y_pred, metric="accuracy", num_labels=3) == 0.75
    assert round(metric_value(y_true, y_pred, metric="macro_f1", num_labels=3), 6) == 0.777778


def test_paired_seed_delta_reports_per_seed_differences() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    model_a = [
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([0, 1, 1, 1], dtype=np.int64),
    ]
    model_b = [
        np.array([0, 0, 0, 1], dtype=np.int64),
        np.array([1, 1, 1, 1], dtype=np.int64),
    ]

    deltas, observed = paired_seed_delta(
        y_true,
        model_a,
        model_b,
        metric="accuracy",
        num_labels=2,
    )

    assert deltas == [0.25, 0.25]
    assert observed == 0.25


def test_bootstrap_and_randomization_return_bounded_statistics() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    model_a = [
        np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        np.array([0, 1, 0, 1, 1, 1], dtype=np.int64),
    ]
    model_b = [
        np.array([0, 0, 0, 1, 0, 1], dtype=np.int64),
        np.array([0, 0, 0, 1, 0, 1], dtype=np.int64),
    ]

    interval = paired_bootstrap_confidence_interval(
        y_true,
        model_a,
        model_b,
        metric="accuracy",
        num_labels=2,
        iterations=200,
        random_state=7,
    )
    test_result = approximate_randomization_test(
        y_true,
        model_a,
        model_b,
        metric="accuracy",
        num_labels=2,
        iterations=200,
        random_state=7,
    )

    assert interval["ci_low"] <= interval["ci_high"]
    assert 0.0 <= test_result["p_value"] <= 1.0
    assert test_result["observed_delta"] > 0.0
