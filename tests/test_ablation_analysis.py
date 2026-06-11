from __future__ import annotations

from reviewguard.analysis.ablation import build_ablation_rows, classify_transfer, significance_note


def test_classify_transfer_distinguishes_gain_drop_and_inconclusive() -> None:
    assert classify_transfer(0.08, 0.02, 0.11, 0.01) == "positive_transfer"
    assert classify_transfer(-0.08, -0.11, -0.02, 0.01) == "negative_transfer"
    assert classify_transfer(0.01, -0.04, 0.06, 0.60) == "inconclusive"


def test_significance_note_formats_human_readable_summary() -> None:
    note = significance_note(0.08, 0.02, 0.11, 0.01)
    assert "significant gain" in note
    assert "p=0.0100" in note


def test_build_ablation_rows_combines_multiseed_and_pairwise_payloads() -> None:
    multiseed_summary = {
        "aggregates": {
            "baseline": {
                "test_metrics": {
                    "sentiment": {"macro_f1": {"mean": 0.73, "std": 0.0}},
                    "authenticity": {"macro_f1": {"mean": 0.80, "std": 0.0}},
                }
            },
            "single-task-sentiment": {
                "test_metrics": {"sentiment": {"macro_f1": {"mean": 0.58, "std": 0.05}}}
            },
            "single-task-authenticity": {
                "test_metrics": {"authenticity": {"macro_f1": {"mean": 0.83, "std": 0.10}}}
            },
            "multitask": {
                "test_metrics": {
                    "sentiment": {"macro_f1": {"mean": 0.59, "std": 0.06}},
                    "authenticity": {"macro_f1": {"mean": 0.91, "std": 0.05}},
                }
            },
        }
    }
    pairwise_rows = [
        {
            "task": "sentiment",
            "metric": "macro_f1",
            "model_a": "multitask",
            "model_b": "single-task-sentiment",
            "observed_delta": 0.004,
            "ci_low": -0.09,
            "ci_high": 0.09,
            "p_value": 0.95,
        },
        {
            "task": "sentiment",
            "metric": "macro_f1",
            "model_a": "multitask",
            "model_b": "baseline",
            "observed_delta": -0.15,
            "ci_low": -0.25,
            "ci_high": -0.04,
            "p_value": 0.002,
        },
        {
            "task": "authenticity",
            "metric": "macro_f1",
            "model_a": "multitask",
            "model_b": "single-task-authenticity",
            "observed_delta": 0.08,
            "ci_low": 0.04,
            "ci_high": 0.12,
            "p_value": 0.0005,
        },
        {
            "task": "authenticity",
            "metric": "macro_f1",
            "model_a": "multitask",
            "model_b": "baseline",
            "observed_delta": 0.11,
            "ci_low": 0.04,
            "ci_high": 0.19,
            "p_value": 0.0005,
        },
    ]

    rows = build_ablation_rows(
        multiseed_summary=multiseed_summary,
        pairwise_rows=pairwise_rows,
    )

    assert len(rows) == 2
    sentiment_row = next(row for row in rows if row["task"] == "sentiment")
    authenticity_row = next(row for row in rows if row["task"] == "authenticity")
    assert sentiment_row["multitask_vs_single_task_verdict"] == "inconclusive"
    assert authenticity_row["multitask_vs_single_task_verdict"] == "positive_transfer"
