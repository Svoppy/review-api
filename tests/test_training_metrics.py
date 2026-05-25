from reviewguard.training.metrics import compute_classification_metrics, compute_multitask_metrics
from reviewguard.training.splits import split_unified_records


def test_compute_classification_metrics_returns_expected_summary() -> None:
    metrics = compute_classification_metrics(
        ["negative", "neutral", "positive", "positive"],
        ["negative", "neutral", "negative", "positive"],
        labels=["negative", "neutral", "positive"],
    )

    assert metrics.support == 4
    assert metrics.accuracy == 0.75
    assert round(metrics.macro_f1, 4) == 0.7778
    assert metrics.per_label_support == {"negative": 1, "neutral": 1, "positive": 2}
    assert metrics.confusion_matrix == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_compute_multitask_metrics_ignores_missing_labels() -> None:
    records = [
        {
            "text": "good",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "bad",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": None,
            "authenticity_label": "fake",
        },
        {
            "text": "ok",
            "source": "b",
            "language": "ru",
            "domain": "ecommerce",
            "sentiment_label": "neutral",
            "authenticity_label": "authentic",
        },
    ]

    metrics = compute_multitask_metrics(
        records,
        sentiment_predictions=["positive", "negative", "neutral"],
        authenticity_predictions=["fake", "fake", "authentic"],
        sentiment_labels=["negative", "neutral", "positive"],
        authenticity_labels=["authentic", "fake"],
    )

    assert metrics["sentiment"]["support"] == 2
    assert metrics["sentiment"]["accuracy"] == 1.0
    assert metrics["authenticity"]["support"] == 2
    assert metrics["authenticity"]["accuracy"] == 1.0


def test_split_unified_records_preserves_all_examples() -> None:
    records = [
        {
            "text": f"review-{index}",
            "source": "demo",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "positive" if index % 2 == 0 else "negative",
            "authenticity_label": "authentic" if index % 2 == 0 else "fake",
        }
        for index in range(10)
    ]

    split = split_unified_records(
        records,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
        random_state=7,
    )

    assert len(split["train"]) == 6
    assert len(split["valid"]) == 2
    assert len(split["test"]) == 2
    assert len(split["train"]) + len(split["valid"]) + len(split["test"]) == len(records)
