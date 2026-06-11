from __future__ import annotations

import json
from pathlib import Path

from reviewguard.analysis.article_tables import build_article_results_markdown


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_article_results_markdown_includes_core_sections(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "rureviews.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    (processed_dir / "perekrestok.jsonl").write_text('{"text":"a"}\n', encoding="utf-8")
    (processed_dir / "maide_up.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n', encoding="utf-8")

    pilot_summary_path = tmp_path / "summary.json"
    _write_json(
        pilot_summary_path,
        {
            "aggregates": {
                "baseline": {},
                "multitask": {},
                "single-task-authenticity": {},
                "single-task-sentiment": {},
            }
        },
    )

    statistics_summary_path = tmp_path / "statistics.json"
    _write_json(
        statistics_summary_path,
        {
            "model_intervals": {
                "baseline": {
                    "test_metrics": {
                        "sentiment": {
                            "accuracy": {"mean": 0.8, "std": 0.0},
                            "macro_f1": {"mean": 0.7, "std": 0.0},
                            "weighted_f1": {"mean": 0.79, "std": 0.0},
                            "precision_macro": {"mean": 0.71, "std": 0.0},
                            "recall_macro": {"mean": 0.72, "std": 0.0},
                        },
                        "authenticity": {
                            "accuracy": {"mean": 0.81, "std": 0.0},
                            "macro_f1": {"mean": 0.8, "std": 0.0},
                            "weighted_f1": {"mean": 0.8, "std": 0.0},
                            "precision_macro": {"mean": 0.8, "std": 0.0},
                            "recall_macro": {"mean": 0.8, "std": 0.0},
                        },
                    }
                },
                "single-task-sentiment": {
                    "test_metrics": {
                        "sentiment": {
                            "accuracy": {"mean": 0.81, "std": 0.01},
                            "macro_f1": {"mean": 0.58, "std": 0.05},
                            "weighted_f1": {"mean": 0.8, "std": 0.01},
                            "precision_macro": {"mean": 0.6, "std": 0.09},
                            "recall_macro": {"mean": 0.58, "std": 0.03},
                        }
                    }
                },
                "single-task-authenticity": {
                    "test_metrics": {
                        "authenticity": {
                            "accuracy": {"mean": 0.84, "std": 0.09},
                            "macro_f1": {"mean": 0.83, "std": 0.1},
                            "weighted_f1": {"mean": 0.83, "std": 0.1},
                            "precision_macro": {"mean": 0.87, "std": 0.04},
                            "recall_macro": {"mean": 0.84, "std": 0.09},
                        }
                    }
                },
                "multitask": {
                    "test_metrics": {
                        "sentiment": {
                            "accuracy": {"mean": 0.73, "std": 0.03},
                            "macro_f1": {"mean": 0.59, "std": 0.07},
                            "weighted_f1": {"mean": 0.72, "std": 0.03},
                            "precision_macro": {"mean": 0.61, "std": 0.1},
                            "recall_macro": {"mean": 0.58, "std": 0.05},
                        },
                        "authenticity": {
                            "accuracy": {"mean": 0.91, "std": 0.05},
                            "macro_f1": {"mean": 0.92, "std": 0.05},
                            "weighted_f1": {"mean": 0.92, "std": 0.05},
                            "precision_macro": {"mean": 0.92, "std": 0.04},
                            "recall_macro": {"mean": 0.91, "std": 0.05},
                        },
                    }
                },
            },
            "pairwise_comparisons": [
                {
                    "task": "authenticity",
                    "model_a": "multitask",
                    "model_b": "single-task-authenticity",
                    "metrics": {
                        "macro_f1": {
                            "observed_delta": 0.08,
                            "bootstrap_ci95": {"ci_low": 0.04, "ci_high": 0.12},
                            "approx_randomization_p_value": 0.0005,
                        }
                    },
                },
                {
                    "task": "authenticity",
                    "model_a": "multitask",
                    "model_b": "baseline",
                    "metrics": {
                        "macro_f1": {
                            "observed_delta": 0.11,
                            "bootstrap_ci95": {"ci_low": 0.05, "ci_high": 0.19},
                            "approx_randomization_p_value": 0.0005,
                        }
                    },
                },
                {
                    "task": "sentiment",
                    "model_a": "multitask",
                    "model_b": "single-task-sentiment",
                    "metrics": {
                        "macro_f1": {
                            "observed_delta": 0.00,
                            "bootstrap_ci95": {"ci_low": -0.09, "ci_high": 0.09},
                            "approx_randomization_p_value": 0.9510,
                        }
                    },
                },
                {
                    "task": "sentiment",
                    "model_a": "multitask",
                    "model_b": "baseline",
                    "metrics": {
                        "macro_f1": {
                            "observed_delta": -0.15,
                            "bootstrap_ci95": {"ci_low": -0.25, "ci_high": -0.04},
                            "approx_randomization_p_value": 0.0025,
                        }
                    },
                },
            ],
        },
    )

    balanced_audit_path = tmp_path / "audit.json"
    _write_json(
        balanced_audit_path,
        {
            "records": 6000,
            "source_distribution": {"rureviews": 2500, "perekrestok": 2500, "maide_up": 1000},
            "duplicate_summary": {"exact_duplicate_rows": 81, "normalized_duplicate_rows": 97},
            "label_coverage": {
                "source": {
                    "authenticity": {
                        "maide_up": {"label_coverage": 1.0},
                        "rureviews": {"label_coverage": 0.0},
                        "perekrestok": {"label_coverage": 0.0},
                    }
                }
            },
            "split_overlap": {
                "train_vs_valid": {"normalized_text_overlap": 0},
                "train_vs_test": {"normalized_text_overlap": 0},
                "valid_vs_test": {"normalized_text_overlap": 0},
            },
            "tasks": {
                "sentiment": {
                    "test_distribution": {
                        "counts": {"negative": 187, "neutral": 8, "positive": 411},
                        "proportions": {"negative": 0.3, "neutral": 0.01, "positive": 0.69},
                    },
                    "majority_baseline": {
                        "test": {"accuracy": 0.67, "macro_f1": 0.26},
                    },
                },
                "authenticity": {
                    "test_distribution": {
                        "counts": {"authentic": 51, "fake": 49},
                        "proportions": {"authentic": 0.51, "fake": 0.49},
                    },
                    "majority_baseline": {
                        "test": {"accuracy": 0.51, "macro_f1": 0.33},
                    },
                },
            },
        },
    )

    baseline_report_path = tmp_path / "baseline_report.json"
    _write_json(
        baseline_report_path,
        {
            "test_metrics": {
                "sentiment": {
                    "confusion_matrix": [[77, 0, 22], [0, 3, 2], [14, 2, 80]],
                }
            }
        },
    )
    multitask_report_path = tmp_path / "multitask_report.json"
    _write_json(
        multitask_report_path,
        {
            "test_metrics": {
                "sentiment": {
                    "confusion_matrix": [[86, 3, 10], [2, 2, 1], [43, 1, 52]],
                },
                "authenticity": {
                    "confusion_matrix": [[37, 13], [1, 49]],
                },
            }
        },
    )
    single_task_sentiment_report_path = tmp_path / "single_task_sentiment_report.json"
    _write_json(
        single_task_sentiment_report_path,
        {
            "test_metrics": {
                "sentiment": {
                    "confusion_matrix": [[89, 0, 10], [4, 0, 1], [24, 0, 72]],
                }
            }
        },
    )
    single_task_authenticity_report_path = tmp_path / "single_task_authenticity_report.json"
    _write_json(
        single_task_authenticity_report_path,
        {
            "test_metrics": {
                "authenticity": {
                    "confusion_matrix": [[23, 27], [0, 50]],
                }
            }
        },
    )

    markdown = build_article_results_markdown(
        processed_dir=processed_dir,
        pilot_summary_path=pilot_summary_path,
        statistics_summary_path=statistics_summary_path,
        balanced_audit_path=balanced_audit_path,
        baseline_report_path=baseline_report_path,
        multitask_report_path=multitask_report_path,
        single_task_sentiment_report_path=single_task_sentiment_report_path,
        single_task_authenticity_report_path=single_task_authenticity_report_path,
    )

    assert "# Пакет итоговых таблиц статьи" in markdown
    assert "## Таблица 2. Основное сравнение моделей" in markdown
    assert "`RuReviews`" in markdown
    assert "лучше single-task и baseline" in markdown
    assert "balanced6k" in markdown
    assert "Representative confusion matrices" in markdown
    assert "single-task sentiment фактически рушится на редком классе `neutral`" in markdown
