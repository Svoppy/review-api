from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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


def _record_identity_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    record_id = record.get("record_id")
    if record_id in (None, ""):
        return None
    source = normalize_source_name(str(record.get("source") or "unknown"))
    return (source, str(record_id))


def validate_normalized_records(records: Iterable[UnifiedReviewRecord | Mapping[str, Any]]) -> dict[str, int]:
    seen_record_ids: set[tuple[str, str]] = set()
    summary = {
        "records": 0,
        "empty_text_records": 0,
        "duplicate_record_ids": 0,
    }
    for record in records:
        payload = record.to_dict() if isinstance(record, UnifiedReviewRecord) else dict(record)
        summary["records"] += 1
        if not normalize_text(payload.get("text")):
            summary["empty_text_records"] += 1
        record_key = _record_identity_key(payload)
        if record_key is not None:
            if record_key in seen_record_ids:
                summary["duplicate_record_ids"] += 1
            seen_record_ids.add(record_key)

    if summary["empty_text_records"] > 0:
        raise ValueError(
            f"Normalized dataset contains {summary['empty_text_records']} records with empty text."
        )
    if summary["duplicate_record_ids"] > 0:
        raise ValueError(
            f"Normalized dataset contains {summary['duplicate_record_ids']} duplicate source-scoped record ids."
        )
    return summary


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


def _resolve_local_export(
    path: str | Path,
    *,
    dataset_name: str,
    candidates: tuple[str, ...],
) -> Path:
    dataset_path = Path(path)
    if dataset_path.is_file():
        return dataset_path
    if dataset_path.is_dir():
        for candidate in candidates:
            candidate_path = dataset_path / candidate
            if candidate_path.is_file():
                return candidate_path
    supported = ", ".join(candidates)
    raise ValueError(
        f"{dataset_name} loader expects a local tabular export file or a directory containing one of: {supported}"
    )


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
        language=normalize_language(language),
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


def normalize_language(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    aliases = {
        "english": "en",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "spanish": "es",
        "turkish": "tr",
        "korean": "ko",
        "romanian": "ro",
        "chinese": "zh",
        "russian": "ru",
    }
    return aliases.get(normalized, normalized)


def load_rureviews(path: str | Path) -> list[UnifiedReviewRecord]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header == "review\tsentiment":
            records: list[UnifiedReviewRecord] = []
            buffer = ""
            record_index = 0
            for line in handle:
                if not line.strip() and not buffer:
                    continue
                buffer += line
                candidate = buffer.rstrip("\n")
                if "\t" not in candidate:
                    continue
                text, sentiment_label = candidate.rsplit("\t", 1)
                try:
                    normalize_sentiment_label(sentiment_label)
                except ValueError:
                    continue
                if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].replace('""', '"')
                record_index += 1
                records.append(
                    _build_record(
                        text=text,
                        source="rureviews",
                        language="ru",
                        domain="ecommerce",
                        sentiment_label=sentiment_label,
                        record_id=record_index,
                        metadata={
                            "raw_dataset": "RuReviews",
                            "sentiment_label_origin": "dataset_label",
                        },
                    )
                )
                buffer = ""
            return records

    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(dataset_path):
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
                metadata={
                    "raw_dataset": "RuReviews",
                    "sentiment_label_origin": "dataset_label",
                },
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
                title=_choose_field(row, "title", "summary", "product_name"),
                product_id=_choose_field(row, "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id", "review_author"),
                rating=rating,
                metadata={
                    "raw_dataset": "Perekrestok-style",
                    "sentiment_label_origin": "rating_heuristic",
                    "product_category": _choose_field(row, "product_category", "category"),
                    "product_price": _choose_field(row, "product_price", "price"),
                },
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
                    record_id=str(file_path.relative_to(dataset_path)),
                    metadata={
                        "raw_dataset": "OpSpam",
                        "path": str(file_path.relative_to(dataset_path)),
                        "sentiment_label_origin": "directory_structure" if sentiment is not None else None,
                        "authenticity_label_origin": "directory_structure",
                        "authenticity_subtype": "crowdsourced_deception",
                    },
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
                metadata={
                    "raw_dataset": "OpSpam",
                    "sentiment_label_origin": "dataset_label",
                    "authenticity_label_origin": "dataset_label",
                    "authenticity_subtype": "crowdsourced_deception",
                },
            )
        )
    return records


def load_maide_up(path: str | Path) -> list[UnifiedReviewRecord]:
    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(path):
        upside_review = _choose_field(row, "Upside_Review", "upside_review", "pros")
        downside_review = _choose_field(row, "Downside_Review", "downside_review", "cons")
        text = _choose_field(row, "text", "review", "review_text", "content")
        if text is None:
            parts: list[str] = []
            if upside_review not in (None, ""):
                parts.append(f"Pros: {upside_review}")
            if downside_review not in (None, ""):
                parts.append(f"Cons: {downside_review}")
            text = "\n\n".join(parts)

        authenticity_value = _choose_field(
            row,
            "authenticity",
            "authenticity_label",
            "review_type",
            "label",
            "source_type",
        )
        authenticity_origin = "dataset_label" if authenticity_value is not None else None
        if authenticity_value is None:
            if "is_ai_generated" in row:
                authenticity_value = (
                    "fake"
                    if str(row["is_ai_generated"]).strip().lower() in {"1", "true", "yes"}
                    else "authentic"
                )
                authenticity_origin = "is_ai_generated_flag"
            elif "source" in row:
                source_value = str(row["source"]).strip().lower()
                if source_value in {"1", "ai", "generated", "synthetic", "llm"}:
                    authenticity_value = "fake"
                    authenticity_origin = "source_field"
                elif source_value in {"0", "human", "real"}:
                    authenticity_value = "authentic"
                    authenticity_origin = "source_field"

        rating = _choose_field(row, "rating", "stars", "score")
        sentiment_value = _choose_field(row, "sentiment", "sentiment_label", "polarity", "Sentiment")
        sentiment_origin = "dataset_label" if sentiment_value is not None else None
        if sentiment_value is None and rating not in (None, ""):
            sentiment_value = map_rating_to_sentiment(rating)
            sentiment_origin = "rating_heuristic"

        hotel_name = _choose_field(row, "Hotel Name", "hotel_name", "title", "summary")
        city_name = _choose_field(row, "City Name", "city_name")
        language = _choose_field(row, "language", "lang", "Review_Language") or "en"
        domain = _choose_field(row, "domain", "category") or ("hospitality" if hotel_name else "ecommerce")
        raw_record_id = _choose_field(row, "id", "review_id", "Unnamed: 0")
        source_suffix = row.get("source")
        record_id = raw_record_id
        if raw_record_id not in (None, "") and source_suffix not in (None, ""):
            record_id = ":".join(
                [
                    str(raw_record_id),
                    str(source_suffix),
                    str(language),
                    str(hotel_name or city_name or ""),
                ]
            )

        records.append(
            _build_record(
                text=text,
                source="maide_up",
                language=language,
                domain=domain,
                sentiment_label=sentiment_value,
                authenticity_label=authenticity_value,
                record_id=record_id,
                title=hotel_name,
                product_id=_choose_field(row, "product_id", "item_id", "sku") or hotel_name,
                user_id=_choose_field(row, "user_id", "author_id"),
                rating=rating or _choose_field(row, "Review_Score"),
                metadata={
                    "raw_dataset": "MAiDE-up-shaped",
                    "sentiment_label_origin": sentiment_origin,
                    "authenticity_label_origin": authenticity_origin,
                    "authenticity_subtype": "ai_generated",
                    "city_name": city_name,
                    "prompt_language": _choose_field(row, "Prompt_Language", "prompt_language"),
                    "na_up_review": row.get("na_up_review"),
                    "na_down_review": row.get("na_down_review"),
                },
            )
        )
    return records


def load_fraudyelp(path: str | Path) -> list[UnifiedReviewRecord]:
    dataset_path = _resolve_local_export(
        path,
        dataset_name="FraudYelpDataset",
        candidates=(
            "reviews.jsonl",
            "reviews.json",
            "reviews.csv",
            "fraudyelp.jsonl",
            "fraudyelp.json",
            "fraudyelp.csv",
            "yelp_reviews.jsonl",
            "yelp_reviews.json",
            "yelp_reviews.csv",
        ),
    )
    records: list[UnifiedReviewRecord] = []
    for row in _iter_tabular_rows(dataset_path):
        text_value = _choose_field(
            row,
            "text",
            "review",
            "review_text",
            "review_content",
            "content",
            "comment",
            "body",
        )
        authenticity_raw = _choose_field(
            row,
            "authenticity",
            "authenticity_label",
            "fraud_label",
            "label",
            "class",
            "y",
            "is_fraud",
        )
        sentiment_raw = _choose_field(row, "sentiment", "sentiment_label", "polarity")
        sentiment_origin = "dataset_label" if sentiment_raw is not None else None
        rating = _choose_field(row, "rating", "stars", "score")
        if sentiment_raw is None and rating not in (None, ""):
            sentiment_raw = map_rating_to_sentiment(rating)
            sentiment_origin = "rating_heuristic"

        records.append(
            _build_record(
                text=text_value,
                source="fraudyelp",
                language=_choose_field(row, "language", "lang") or "en",
                domain=_choose_field(row, "domain", "category") or "local_commerce",
                sentiment_label=sentiment_raw,
                authenticity_label=authenticity_raw,
                record_id=_choose_field(row, "review_id", "id", "record_id"),
                title=_choose_field(row, "title", "summary", "business_name"),
                product_id=_choose_field(row, "business_id", "product_id", "item_id", "sku"),
                user_id=_choose_field(row, "user_id", "author_id"),
                rating=rating,
                metadata={
                    "raw_dataset": "FraudYelpDataset-shaped",
                    "sentiment_label_origin": sentiment_origin,
                    "authenticity_label_origin": "dataset_label",
                    "authenticity_subtype": "silver_fraud",
                    "raw_authenticity_label": None if authenticity_raw in (None, "") else str(authenticity_raw),
                    "raw_sentiment_label": None if sentiment_origin != "dataset_label" or sentiment_raw in (None, "") else str(sentiment_raw),
                    "raw_export_file": dataset_path.name,
                    "split": _choose_field(row, "split", "partition"),
                },
            )
        )
    return records


DATASET_LOADERS = {
    "rureviews": load_rureviews,
    "perekrestok": load_perekrestok_ratings,
    "opspam": load_opspam,
    "maide_up": load_maide_up,
    "fraudyelp": load_fraudyelp,
    "fraud_yelp": load_fraudyelp,
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
    records = DATASET_LOADERS[normalized_name](path)
    validate_normalized_records(records)
    return records


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


def merge_unified_datasets(
    input_paths: Iterable[str | Path],
    *,
    require_any_label: bool = True,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    indexed_records: dict[tuple[str, ...], dict[str, Any]] = {}

    for input_path in input_paths:
        for record in load_unified_records(input_path):
            if require_any_label and not (
                record.get("sentiment_label") is not None
                or record.get("authenticity_label") is not None
            ):
                continue

            dedupe_key = stable_record_identity(record)
            if dedupe_key is None:
                merged.append(record)
                continue

            existing = indexed_records.get(dedupe_key)
            if existing is None:
                indexed_records[dedupe_key] = record
                merged.append(record)
                continue

            merge_record_labels(existing, record)

    return merged


def stable_record_identity(record: Mapping[str, Any]) -> tuple[str, ...] | None:
    source = str(record.get("source") or "unknown")
    record_id = record.get("record_id")
    if record_id not in (None, ""):
        return ("record_id", source, str(record_id))

    product_id = record.get("product_id")
    user_id = record.get("user_id")
    rating = record.get("rating")
    title = record.get("title")

    # Without a stronger identity signal, keep repeated same-text rows as distinct examples.
    if all(value in (None, "") for value in (product_id, user_id, rating, title)):
        return None

    return (
        "composite",
        source,
        str(product_id or ""),
        str(user_id or ""),
        str(rating or ""),
        str(title or ""),
        str(record.get("text") or ""),
    )


def merge_record_labels(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    record_hint = existing.get("record_id") or incoming.get("record_id") or existing.get("text") or "<unknown>"

    for field in ("text", "title", "product_id", "user_id", "rating", "language", "domain"):
        current = existing.get(field)
        update = incoming.get(field)
        if current not in (None, "") and update not in (None, "") and current != update:
            raise ValueError(
                f"Conflicting {field} values for merged record {record_hint!r}: {current!r} vs {update!r}"
            )

    for label_field in ("sentiment_label", "authenticity_label"):
        current = existing.get(label_field)
        update = incoming.get(label_field)
        if current is not None and update is not None and current != update:
            raise ValueError(
                f"Conflicting {label_field} values for merged record {record_hint!r}: {current!r} vs {update!r}"
            )
        if current is None and update is not None:
            existing[label_field] = update

    incoming_metadata = incoming.get("metadata")
    if isinstance(incoming_metadata, dict):
        existing_metadata = existing.setdefault("metadata", {})
        if isinstance(existing_metadata, dict):
            existing_metadata.update(incoming_metadata)

    for field in ("product_id", "user_id", "rating", "title"):
        if existing.get(field) in (None, "") and incoming.get(field) not in (None, ""):
            existing[field] = incoming[field]


def summarize_unified_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": 0,
        "sentiment_labeled": 0,
        "authenticity_labeled": 0,
        "sources": {},
        "languages": {},
    }
    for record in records:
        summary["records"] += 1
        if record.get("sentiment_label") is not None:
            summary["sentiment_labeled"] += 1
        if record.get("authenticity_label") is not None:
            summary["authenticity_labeled"] += 1

        source = str(record.get("source") or "unknown")
        language = str(record.get("language") or "unknown")
        summary["sources"][source] = summary["sources"].get(source, 0) + 1
        summary["languages"][language] = summary["languages"].get(language, 0) + 1

    return summary


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
    subparsers = parser.add_subparsers(dest="command")

    normalize = subparsers.add_parser("normalize", help="Normalize one raw dataset into the unified schema.")
    normalize.add_argument("--dataset", required=True, choices=sorted(DATASET_LOADERS))
    normalize.add_argument("--input", required=True, dest="input_path")
    normalize.add_argument("--output", required=True, dest="output_path")
    normalize.add_argument("--format", choices=("jsonl", "csv"), dest="output_format")

    merge = subparsers.add_parser("merge", help="Merge multiple unified dataset files into one joint corpus.")
    merge.add_argument("--inputs", nargs="+", required=True, dest="input_paths")
    merge.add_argument("--output", required=True, dest="output_path")
    merge.add_argument("--format", choices=("jsonl", "csv"), default="jsonl", dest="output_format")
    merge.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Keep records even if both task labels are missing.",
    )

    audit = subparsers.add_parser(
        "audit",
        help="Create a split-level audit report with class balance, majority baseline, and overlap checks.",
    )
    audit.add_argument("--input", required=True, dest="input_path")
    audit.add_argument("--output", required=True, dest="output_path")
    audit.add_argument("--train-size", type=float, default=0.8)
    audit.add_argument("--valid-size", type=float, default=0.1)
    audit.add_argument("--test-size", type=float, default=0.1)
    audit.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser = build_arg_parser()
        parser.print_help()
        return 0

    if argv[0] not in {"normalize", "merge", "audit"} and argv[0].startswith("-") and argv[0] not in {"-h", "--help"}:
        argv = ["normalize", *argv]

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command == "normalize":
        process_dataset(
            dataset=args.dataset,
            input_path=args.input_path,
            output_path=args.output_path,
            output_format=args.output_format,
        )
        return 0

    if command == "merge":
        records = merge_unified_datasets(
            args.input_paths,
            require_any_label=not args.allow_unlabeled,
        )
        if args.output_format == "csv":
            write_csv(args.output_path, [_build_record(**record) for record in records])
        else:
            write_jsonl(args.output_path, [_build_record(**record) for record in records])
        return 0

    if command == "audit":
        from reviewguard.data.audit import build_dataset_audit_report, write_audit_report
        from reviewguard.training.splits import split_unified_records

        records = load_unified_records(args.input_path)
        split = split_unified_records(
            records,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        report = build_dataset_audit_report(
            records,
            split,
            input_path=args.input_path,
            random_state=args.random_state,
        )
        write_audit_report(args.output_path, report)
        return 0

    parser.print_help()
    return 0
