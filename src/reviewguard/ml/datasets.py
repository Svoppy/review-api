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
        name="rureviews",
        task="sentiment",
        source_url="https://github.com/sismetanin/rureviews",
        description="Russian product-review benchmark with direct sentiment labels.",
        caveat="Covers sentiment only and does not provide authenticity annotation.",
    ),
    DatasetSpec(
        name="perekrestok",
        task="sentiment",
        source_url="https://huggingface.co/datasets/lapki/perekrestok-reviews",
        description="Large in-domain Russian retail review corpus with rating-derived sentiment.",
        caveat="Sentiment is derived from ratings rather than directly annotated for polarity.",
    ),
    DatasetSpec(
        name="opspam",
        task="authenticity",
        source_url="https://myleott.com/op-spam.html",
        description="Text-only deceptive opinion benchmark.",
        caveat="Collected in an elicited experimental setup.",
    ),
    DatasetSpec(
        name="fraudyelp",
        task="authenticity",
        source_url="https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudDataset.html",
        description="Silver-label fraud benchmark that requires a local review-text export for this repository pipeline.",
        caveat="Labels are proxy fraud labels and should not be treated as perfect review-level ground truth.",
    ),
    DatasetSpec(
        name="maide_up",
        task="authenticity",
        source_url="https://huggingface.co/datasets/MichiganNLP/MAiDE-up",
        description="Multilingual benchmark for detecting AI-generated fake reviews.",
        caveat="Targets synthetic-review authenticity rather than the full spectrum of platform fraud.",
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
