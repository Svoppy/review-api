from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runner_module():
    module_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_multiseed_experiments.py"
    )
    spec = importlib.util.spec_from_file_location("run_multiseed_experiments", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_runs_computes_mean_and_std_for_task_metrics() -> None:
    runner = _load_runner_module()
    runs = [
        {
            "model_id": "multitask",
            "train_seed": 11,
            "split_seed": 42,
            "report": {
                "test_metrics": {
                    "sentiment": {
                        "accuracy": 0.70,
                        "macro_f1": 0.68,
                        "support": 200,
                    }
                },
                "validation_metrics": {},
            },
        },
        {
            "model_id": "multitask",
            "train_seed": 21,
            "split_seed": 42,
            "report": {
                "test_metrics": {
                    "sentiment": {
                        "accuracy": 0.80,
                        "macro_f1": 0.72,
                        "support": 200,
                    }
                },
                "validation_metrics": {},
            },
        },
    ]

    aggregates = runner.aggregate_runs(runs)
    metrics = aggregates["multitask"]["test_metrics"]["sentiment"]

    assert metrics["accuracy"]["n"] == 2
    assert metrics["accuracy"]["mean"] == 0.75
    assert round(metrics["accuracy"]["std"], 6) == 0.070711
    assert metrics["macro_f1"]["values"] == [0.68, 0.72]


def test_build_markdown_summary_formats_test_metrics_table() -> None:
    runner = _load_runner_module()
    markdown = runner.build_markdown_summary(
        input_path=Path("data/processed/joint_reviews.pilot1k.jsonl"),
        split_seed=42,
        train_seeds=[11, 21, 42],
        aggregates={
            "multitask": {
                "test_metrics": {
                    "authenticity": {
                        "macro_f1": {"mean": 0.8123, "std": 0.0142},
                        "accuracy": {"mean": 0.8300, "std": 0.0100},
                    }
                },
                "validation_metrics": {},
            }
        },
    )

    assert "# Multi-Seed Experiment Summary" in markdown
    assert "`multitask`" in markdown
    assert "0.8123 +/- 0.0142" in markdown
