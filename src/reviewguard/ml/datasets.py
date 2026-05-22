from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SENTIMENT_LABELS = ["negative", "neutral", "positive"]
AUTHENTICITY_LABELS = ["authentic", "fake"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    source_url: str
    description: str
    caveat: str


DATASET_REGISTRY = [
    DatasetSpec(
        name="amazon_reviews_2023",
        task="sentiment",
        source_url="https://amazon-reviews-2023.github.io/main.html",
        description="Large-scale real Amazon review corpus for product-review sentiment.",
        caveat="Sentiment is typically derived from ratings, not manually annotated sentence labels.",
    ),
    DatasetSpec(
        name="rusentiment",
        task="sentiment",
        source_url="https://rusentiment.github.io/",
        description="Russian sentiment benchmark useful for multilingual validation.",
        caveat="Not e-commerce specific.",
    ),
    DatasetSpec(
        name="opspam",
        task="authenticity",
        source_url="https://myleott.com/op-spam.html",
        description="Text-only deceptive opinion benchmark.",
        caveat="Collected in an elicited experimental setup.",
    ),
    DatasetSpec(
        name="yelpchi",
        task="authenticity",
        source_url="https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.922589/full",
        description="Widely used fake review detection benchmark.",
        caveat="Labels are benchmark labels with real-world ambiguity.",
    ),
]


def map_star_rating_to_sentiment(rating: int | float) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def build_multitask_record(
    *,
    text: str,
    source: str,
    language: str,
    domain: str,
    sentiment_label: str | None = None,
    authenticity_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "source": source,
        "language": language,
        "domain": domain,
        "sentiment_label": sentiment_label,
        "authenticity_label": authenticity_label,
        "metadata": metadata or {},
    }
