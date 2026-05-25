from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.model_selection import train_test_split


def _stratify_key(record: dict[str, Any]) -> str:
    sentiment = record.get("sentiment_label") or "unlabeled"
    authenticity = record.get("authenticity_label") or "unlabeled"
    source = record.get("source") or "unknown"
    return f"{source}|s:{sentiment}|a:{authenticity}"


def _should_stratify(records: list[dict[str, Any]]) -> bool:
    if len(records) < 2:
        return False
    counts = Counter(_stratify_key(record) for record in records)
    return min(counts.values(), default=0) >= 2


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

    stratify = [_stratify_key(record) for record in records] if _should_stratify(records) else None

    train_records, holdout_records = train_test_split(
        records,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

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
    holdout_stratify = (
        [_stratify_key(record) for record in holdout_records] if _should_stratify(holdout_records) else None
    )
    valid_records, test_records = train_test_split(
        holdout_records,
        train_size=valid_share_of_holdout,
        random_state=random_state,
        stratify=holdout_stratify,
    )

    return {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }
