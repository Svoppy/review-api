from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SENTIMENT_LABELS = ("negative", "neutral", "positive")
AUTHENTICITY_LABELS = ("authentic", "fake")


@dataclass(slots=True)
class UnifiedReviewRecord:
    text: str
    source: str
    language: str
    domain: str
    sentiment_label: str | None = None
    authenticity_label: str | None = None
    record_id: str | None = None
    title: str | None = None
    product_id: str | None = None
    user_id: str | None = None
    rating: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_source_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_rating(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def map_rating_to_sentiment(value: Any) -> str:
    rating = parse_rating(value)
    if rating is None:
        raise ValueError(f"Cannot derive sentiment from rating: {value!r}")
    if rating <= 2:
        return "negative"
    if rating < 4:
        return "neutral"
    return "positive"


def normalize_sentiment_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return map_rating_to_sentiment(value)

    raw = str(value).strip().lower()
    if raw in {"negative", "neg", "-1", "bad", "anger", "sadness"}:
        return "negative"
    if raw in {"neutral", "neu", "0", "mixed"}:
        return "neutral"
    if raw in {"positive", "pos", "1", "good", "joy"}:
        return "positive"
    if raw in {"negative", "neutral", "positive"}:
        return raw

    rating = parse_rating(raw)
    if rating is not None:
        return map_rating_to_sentiment(rating)
    raise ValueError(f"Unsupported sentiment label: {value!r}")


def normalize_authenticity_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "fake" if value else "authentic"

    raw = str(value).strip().lower()
    if raw in {"authentic", "genuine", "real", "truthful", "human", "0", "false"}:
        return "authentic"
    if raw in {
        "fake",
        "deceptive",
        "spam",
        "fraud",
        "ai",
        "ai_generated",
        "generated",
        "llm",
        "1",
        "true",
    }:
        return "fake"
    raise ValueError(f"Unsupported authenticity label: {value!r}")


def _choose_field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _iter_tabular_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                yield dict(row)
        return

    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for row in payload:
                yield dict(row)
            return
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("rows") or payload.get("reviews")
            if isinstance(rows, list):
                for row in rows:
                    yield dict(row)
                return
        raise ValueError(f"Unsupported JSON dataset shape in {dataset_path}")

    raise ValueError(f"Unsupported file format: {dataset_path.suffix}")


def _build_record(
    *,
    text: Any,
    source: str,
    language: str,
    domain: str,
    sentiment_label: Any = None,
    authenticity_label: Any = None,
    record_id: Any = None,
    title: Any = None,
    product_id: Any = None,
    user_id: Any = None,
    rating: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> UnifiedReviewRecord:
    normalized_rating = parse_rating(rating)
    return UnifiedReviewRecord(
        text=normalize_text(text),
        source=normalize_source_name(source),
        language=str(language).strip().lower(),
        domain=str(domain).strip().lower(),
        sentiment_label=normalize_sentiment_label(sentiment_label),
        authenticity_label=normalize_authenticity_label(authenticity_label),
        record_id=None if record_id in (None, "") else str(record_id),
        title=None if title in (None, "") else normalize_text(title),
        product_id=None if product_id in (None, "") else str(product_id),
        user_id=None if user_id in (None, "") else str(user_id),
        rating=normalized_rating,
        metadata=dict(metadata or {}),
    )


def load_rureviews(path: str | Path) -> list[UnifiedReviewRecord]:
    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(path):
        records.append(
            _build_record(
                text=_choose_field(row, "text", "review", "review_text", "content"),
                source="rureviews",
                language="ru",
                domain="ecommerce",
                sentiment_label=_choose_field(row, "label", "sentiment", "sentiment_label", "class"),
                record_id=_choose_field(row, "id", "review_id"),
                title=_choose_field(row, "title", "summary"),
                product_id=_choose_field(row, "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id"),
                metadata={"raw_dataset": "RuReviews"},
            )
        )
    return records


def load_perekrestok_ratings(path: str | Path) -> list[UnifiedReviewRecord]:
    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(path):
        rating = _choose_field(row, "rating", "stars", "score", "grade")
        records.append(
            _build_record(
                text=_choose_field(row, "text", "review", "review_text", "comment", "body"),
                source="perekrestok",
                language="ru",
                domain="ecommerce",
                sentiment_label=map_rating_to_sentiment(rating),
                record_id=_choose_field(row, "id", "review_id"),
                title=_choose_field(row, "title", "summary"),
                product_id=_choose_field(row, "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id"),
                rating=rating,
                metadata={"raw_dataset": "Perekrestok-style"},
            )
        )
    return records


def load_opspam(path: str | Path) -> list[UnifiedReviewRecord]:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        records: list[UnifiedReviewRecord] = []
        for file_path in sorted(dataset_path.rglob("*.txt")):
            parts = {part.lower() for part in file_path.parts}
            authenticity = "fake" if "deceptive" in parts else "authentic"
            sentiment = "positive" if "positive" in parts else "negative" if "negative" in parts else None
            records.append(
                _build_record(
                    text=file_path.read_text(encoding="utf-8"),
                    source="opspam",
                    language="en",
                    domain="hospitality",
                    sentiment_label=sentiment,
                    authenticity_label=authenticity,
                    record_id=file_path.stem,
                    metadata={"raw_dataset": "OpSpam", "path": str(file_path.relative_to(dataset_path))},
                )
            )
        return records

    records = []
    for row in _iter_tabular_rows(dataset_path):
        records.append(
            _build_record(
                text=_choose_field(row, "text", "review", "review_text", "content"),
                source="opspam",
                language="en",
                domain="hospitality",
                sentiment_label=_choose_field(row, "sentiment", "sentiment_label", "polarity"),
                authenticity_label=_choose_field(row, "authenticity", "label", "deceptive", "class"),
                record_id=_choose_field(row, "id", "review_id"),
                metadata={"raw_dataset": "OpSpam"},
            )
        )
    return records


def load_maide_up(path: str | Path) -> list[UnifiedReviewRecord]:
    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(path):
        authenticity_value = _choose_field(
            row,
            "authenticity",
            "authenticity_label",
            "review_type",
            "label",
            "source_type",
        )
        if authenticity_value is None and "is_ai_generated" in row:
            authenticity_value = "fake" if str(row["is_ai_generated"]).strip().lower() in {"1", "true", "yes"} else "authentic"

        rating = _choose_field(row, "rating", "stars", "score")
        sentiment_value = _choose_field(row, "sentiment", "sentiment_label", "polarity")
        if sentiment_value is None and rating not in (None, ""):
            sentiment_value = map_rating_to_sentiment(rating)

        records.append(
            _build_record(
                text=_choose_field(row, "text", "review", "review_text", "content"),
                source="maide_up",
                language=_choose_field(row, "language", "lang") or "en",
                domain=_choose_field(row, "domain", "category") or "ecommerce",
                sentiment_label=sentiment_value,
                authenticity_label=authenticity_value,
                record_id=_choose_field(row, "id", "review_id"),
                title=_choose_field(row, "title", "summary"),
                product_id=_choose_field(row, "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id"),
                rating=rating,
                metadata={"raw_dataset": "MAiDE-up-shaped"},
            )
        )
    return records


DATASET_LOADERS = {
    "rureviews": load_rureviews,
    "perekrestok": load_perekrestok_ratings,
    "opspam": load_opspam,
    "maide_up": load_maide_up,
}


def write_jsonl(path: str | Path, records: Iterable[UnifiedReviewRecord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def write_csv(path: str | Path, records: Iterable[UnifiedReviewRecord]) -> Path:
    output_path = Path(path)
    rows = [record.to_dict() for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "source",
        "language",
        "domain",
        "title",
        "product_id",
        "user_id",
        "rating",
        "sentiment_label",
        "authenticity_label",
        "text",
        "metadata",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    return output_path


def load_dataset(name: str, path: str | Path) -> list[UnifiedReviewRecord]:
    normalized_name = normalize_source_name(name)
    if normalized_name not in DATASET_LOADERS:
        supported = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unsupported dataset {name!r}. Expected one of: {supported}")
    return DATASET_LOADERS[normalized_name](path)


def load_unified_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _iter_tabular_rows(path):
        metadata = row.get("metadata", {})
        if isinstance(metadata, str) and metadata.strip():
            metadata = json.loads(metadata)
        elif not isinstance(metadata, dict):
            metadata = {}

        records.append(
            _build_record(
                text=_choose_field(row, "text", "review", "review_text", "content"),
                source=_choose_field(row, "source") or "unknown",
                language=_choose_field(row, "language") or "unknown",
                domain=_choose_field(row, "domain") or "unknown",
                sentiment_label=_choose_field(row, "sentiment_label", "sentiment"),
                authenticity_label=_choose_field(row, "authenticity_label", "authenticity"),
                record_id=_choose_field(row, "record_id", "id", "review_id"),
                title=_choose_field(row, "title", "summary"),
                product_id=_choose_field(row, "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id"),
                rating=_choose_field(row, "rating", "stars", "score"),
                metadata=metadata,
            ).to_dict()
        )
    return records


def process_dataset(
    *,
    dataset: str,
    input_path: str | Path,
    output_path: str | Path,
    output_format: str | None = None,
) -> Path:
    records = load_dataset(dataset, input_path)
    destination = Path(output_path)
    file_format = (output_format or destination.suffix.lstrip(".") or "jsonl").lower()
    if file_format == "jsonl":
        return write_jsonl(destination, records)
    if file_format == "csv":
        return write_csv(destination, records)
    raise ValueError(f"Unsupported output format: {file_format}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw review datasets into a unified multitask schema.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LOADERS))
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--format", choices=("jsonl", "csv"), dest="output_format")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    process_dataset(
        dataset=args.dataset,
        input_path=args.input_path,
        output_path=args.output_path,
        output_format=args.output_format,
    )
    return 0
