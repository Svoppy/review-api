from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationComparison:
    task: str
    metric: str
    model_a: str
    model_b: str
    observed_delta: float
    ci_low: float
    ci_high: float
    p_value: float


def classify_transfer(delta: float, ci_low: float, ci_high: float, p_value: float) -> str:
    if p_value < 0.05 and ci_low > 0 and ci_high > 0:
        return "positive_transfer"
    if p_value < 0.05 and ci_low < 0 and ci_high < 0:
        return "negative_transfer"
    return "inconclusive"


def significance_note(delta: float, ci_low: float, ci_high: float, p_value: float) -> str:
    verdict = classify_transfer(delta, ci_low, ci_high, p_value)
    if verdict == "positive_transfer":
        return f"significant gain (p={p_value:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])"
    if verdict == "negative_transfer":
        return f"significant drop (p={p_value:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])"
    return f"no clear difference (p={p_value:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])"


def index_pairwise_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["task"]),
            str(row["metric"]),
            str(row["model_a"]),
            str(row["model_b"]),
        ): row
        for row in rows
    }


def build_ablation_rows(
    *,
    multiseed_summary: dict[str, Any],
    pairwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairwise_index = index_pairwise_rows(pairwise_rows)
    rows: list[dict[str, Any]] = []

    task_to_single_task_model = {
        "sentiment": "single-task-sentiment",
        "authenticity": "single-task-authenticity",
    }

    for task, single_task_model in task_to_single_task_model.items():
        baseline_metrics = multiseed_summary["aggregates"]["baseline"]["test_metrics"][task]
        single_task_metrics = multiseed_summary["aggregates"][single_task_model]["test_metrics"][task]
        multitask_metrics = multiseed_summary["aggregates"]["multitask"]["test_metrics"][task]

        pair_vs_single = pairwise_index[(task, "macro_f1", "multitask", single_task_model)]
        pair_vs_baseline = pairwise_index[(task, "macro_f1", "multitask", "baseline")]

        rows.append(
            {
                "task": task,
                "baseline_macro_f1_mean": baseline_metrics["macro_f1"]["mean"],
                "baseline_macro_f1_std": baseline_metrics["macro_f1"]["std"],
                "single_task_macro_f1_mean": single_task_metrics["macro_f1"]["mean"],
                "single_task_macro_f1_std": single_task_metrics["macro_f1"]["std"],
                "multitask_macro_f1_mean": multitask_metrics["macro_f1"]["mean"],
                "multitask_macro_f1_std": multitask_metrics["macro_f1"]["std"],
                "multitask_vs_single_task_delta": float(pair_vs_single["observed_delta"]),
                "multitask_vs_single_task_ci_low": float(pair_vs_single["ci_low"]),
                "multitask_vs_single_task_ci_high": float(pair_vs_single["ci_high"]),
                "multitask_vs_single_task_p_value": float(pair_vs_single["p_value"]),
                "multitask_vs_single_task_verdict": classify_transfer(
                    float(pair_vs_single["observed_delta"]),
                    float(pair_vs_single["ci_low"]),
                    float(pair_vs_single["ci_high"]),
                    float(pair_vs_single["p_value"]),
                ),
                "multitask_vs_single_task_note": significance_note(
                    float(pair_vs_single["observed_delta"]),
                    float(pair_vs_single["ci_low"]),
                    float(pair_vs_single["ci_high"]),
                    float(pair_vs_single["p_value"]),
                ),
                "multitask_vs_baseline_delta": float(pair_vs_baseline["observed_delta"]),
                "multitask_vs_baseline_ci_low": float(pair_vs_baseline["ci_low"]),
                "multitask_vs_baseline_ci_high": float(pair_vs_baseline["ci_high"]),
                "multitask_vs_baseline_p_value": float(pair_vs_baseline["p_value"]),
                "multitask_vs_baseline_verdict": classify_transfer(
                    float(pair_vs_baseline["observed_delta"]),
                    float(pair_vs_baseline["ci_low"]),
                    float(pair_vs_baseline["ci_high"]),
                    float(pair_vs_baseline["p_value"]),
                ),
                "multitask_vs_baseline_note": significance_note(
                    float(pair_vs_baseline["observed_delta"]),
                    float(pair_vs_baseline["ci_low"]),
                    float(pair_vs_baseline["ci_high"]),
                    float(pair_vs_baseline["p_value"]),
                ),
            }
        )

    return rows

