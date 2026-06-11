from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

from sklearn.model_selection import train_test_split


def _normalized_text(record: dict[str, Any]) -> str:
    text = str(record.get("text") or "")
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _group_key(record: dict[str, Any], index: int) -> str:
    normalized_text = _normalized_text(record)
    if normalized_text:
        fingerprint = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
        # Group exact normalized-text duplicates globally so the same review body
        # cannot leak across splits just because it arrived through another source.
        return f"text:{fingerprint}"

    source = str(record.get("source") or "unknown")

    record_id = record.get("record_id")
    if record_id not in (None, ""):
        return f"{source}|id:{record_id}"

    product_id = record.get("product_id")
    title = record.get("title")
    if product_id not in (None, "") or title not in (None, ""):
        return f"{source}|product:{product_id or ''}|title:{title or ''}"

    return f"{source}|row:{index}"


def _stratify_key(record: dict[str, Any]) -> str:
    sentiment = record.get("sentiment_label") or "unlabeled"
    authenticity = record.get("authenticity_label") or "unlabeled"
    source = record.get("source") or "unknown"
    return f"{source}|s:{sentiment}|a:{authenticity}"


def _should_stratify(keys: list[str]) -> bool:
    if len(keys) < 2:
        return False
    counts = Counter(keys)
    return min(counts.values(), default=0) >= 2


def _group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for index, record in enumerate(records):
        key = _group_key(record, index)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record)
    return [grouped[key] for key in order]


def split_unified_records(
    records: list[dict[str, Any]],
    *,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    total = train_size + valid_size + test_size
    if not records:
        return {"train": [], "valid": [], "test": []}
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size, valid_size, and test_size must sum to 1.0.")
    if min(train_size, valid_size, test_size) < 0:
        raise ValueError("Split sizes must be non-negative.")

    groups = _group_records(records)
    representatives = [group[0] for group in groups]
    stratify_keys = [_stratify_key(record) for record in representatives]
    stratify = stratify_keys if _should_stratify(stratify_keys) else None

    train_groups, holdout_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_records = [record for group in train_groups for record in group]
    holdout_records = [record for group in holdout_groups for record in group]

    if not holdout_records:
        return {"train": train_records, "valid": [], "test": []}

    holdout_fraction = valid_size + test_size
    if holdout_fraction == 0:
        return {"train": train_records, "valid": [], "test": []}
    if valid_size == 0:
        return {"train": train_records, "valid": [], "test": holdout_records}
    if test_size == 0:
        return {"train": train_records, "valid": holdout_records, "test": []}

    valid_share_of_holdout = valid_size / holdout_fraction
    holdout_representatives = [group[0] for group in holdout_groups]
    holdout_keys = [_stratify_key(record) for record in holdout_representatives]
    holdout_stratify = holdout_keys if _should_stratify(holdout_keys) else None
    valid_groups, test_groups = train_test_split(
        holdout_groups,
        train_size=valid_share_of_holdout,
        random_state=random_state,
        stratify=holdout_stratify,
    )
    valid_records = [record for group in valid_groups for record in group]
    test_records = [record for group in test_groups for record in group]

    return {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }
