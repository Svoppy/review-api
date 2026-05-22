from reviewguard.ml.datasets import build_multitask_record, map_star_rating_to_sentiment


def test_map_star_rating_to_sentiment() -> None:
    assert map_star_rating_to_sentiment(1) == "negative"
    assert map_star_rating_to_sentiment(3) == "neutral"
    assert map_star_rating_to_sentiment(5) == "positive"


def test_build_multitask_record_keeps_optional_labels() -> None:
    record = build_multitask_record(
        text="Товар отличный, пришел быстро.",
        source="rureviews",
        language="ru",
        domain="ecommerce",
        sentiment_label="positive",
        authenticity_label=None,
    )
    assert record["sentiment_label"] == "positive"
    assert record["authenticity_label"] is None
