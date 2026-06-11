from __future__ import annotations

from reviewguard.analysis.robustness import build_multitask_robustness_report, build_slice_metrics
from reviewguard.data.audit import build_dataset_audit_report
from reviewguard.training.splits import split_unified_records


def test_build_slice_metrics_reports_gap_across_sources() -> None:
    records = [
        {
            "text": "great",
            "source": "ru",
            "language": "ru",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "excellent",
            "source": "ru",
            "language": "ru",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "awful",
            "source": "hotel",
            "language": "en",
            "domain": "hospitality",
            "sentiment_label": "negative",
            "authenticity_label": None,
        },
        {
            "text": "terrible",
            "source": "hotel",
            "language": "en",
            "domain": "hospitality",
            "sentiment_label": "negative",
            "authenticity_label": None,
        },
    ]

    report = build_slice_metrics(
        records,
        predictions=["positive", "positive", "positive", "negative"],
        task="sentiment",
        labels=["negative", "neutral", "positive"],
        slice_field="source",
        min_support=2,
    )

    assert report["evaluated_slices"] == 2
    assert report["robustness_gap"] is not None
    assert report["robustness_gap"] > 0.0


def test_build_multitask_robustness_report_handles_missing_task_predictions() -> None:
    records = [
        {
            "text": "great",
            "source": "ru",
            "language": "ru",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "fake",
            "source": "maide",
            "language": "en",
            "domain": "hospitality",
            "sentiment_label": None,
            "authenticity_label": "fake",
        },
    ]

    report = build_multitask_robustness_report(
        records,
        sentiment_predictions=["positive", "negative"],
        authenticity_predictions=["authentic", "fake"],
        sentiment_labels=["negative", "neutral", "positive"],
        authenticity_labels=["authentic", "fake"],
    )

    assert "sentiment" in report
    assert "authenticity" in report


def test_dataset_audit_surfaces_small_sample_warning() -> None:
    records = [
        {
            "record_id": str(index),
            "text": f"review-{index}",
            "source": "rureviews" if index < 6 else "maide_up",
            "language": "ru" if index < 6 else "en",
            "domain": "ecommerce" if index < 6 else "hospitality",
            "sentiment_label": "neutral" if index == 0 else "positive",
            "authenticity_label": "fake" if index >= 6 else None,
        }
        for index in range(8)
    ]

    split = split_unified_records(records, train_size=0.5, valid_size=0.25, test_size=0.25, random_state=5)
    report = build_dataset_audit_report(records, split, input_path="demo.jsonl", random_state=5)

    assert report["warnings"]
    assert any("pilot-scale" in warning or "macro-F1 is likely unstable" in warning for warning in report["tasks"]["sentiment"]["warnings"])
