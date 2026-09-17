from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.data import build_dataset_audit_report, load_unified_records, write_audit_report
from reviewguard.training.splits import split_unified_records


DEFAULT_SOURCE_INPUTS = {
    "rureviews": Path("data/processed/rureviews.jsonl"),
    "perekrestok": Path("data/processed/perekrestok.jsonl"),
    "maide_up": Path("data/processed/maide_up.jsonl"),
    "wildberries": Path("data/processed/wildberries.jsonl"),
}

DEFAULT_SOURCE_TARGETS = [
    "perekrestok=10000",
    "rureviews=3000",
    "maide_up=7000",
]

DEFAULT_SOURCE_REQUIRES = [
    "perekrestok:sentiment_label:neutral=1200",
    "maide_up:authenticity_label:authentic=3500",
    "maide_up:authenticity_label:fake=3500",
]


def parse_key_value_pairs(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw_value in values:
        name, raw_count = raw_value.split("=", 1)
        parsed[name] = int(raw_count)
    return parsed


def parse_source_requirements(values: list[str]) -> dict[str, list[tuple[str, str, int]]]:
    requirements: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for raw_value in values:
        left, raw_count = raw_value.split("=", 1)
        source, field, value = left.split(":", 2)
        requirements[source].append((field, value, int(raw_count)))
    return requirements


def load_sources(source_inputs: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    return {name: load_unified_records(path) for name, path in source_inputs.items()}


def sample_source_records(
    records: list[dict[str, Any]],
    *,
    target_size: int,
    requirements: list[tuple[str, str, int]],
    seed: int,
) -> list[dict[str, Any]]:
    if target_size <= 0:
        return []

    rng = random.Random(seed)
    candidates = list(records)
    rng.shuffle(candidates)

    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    for field, value, count in requirements:
        matches = [record for record in candidates if str(record.get(field)) == value and id(record) not in used_ids]
        if len(matches) < count:
            raise ValueError(
                f"Requirement {field}={value}:{count} cannot be satisfied; only {len(matches)} rows available."
            )
        selected.extend(matches[:count])
        used_ids.update(id(record) for record in matches[:count])

    if len(selected) > target_size:
        raise ValueError(
            f"Required rows ({len(selected)}) exceed target_size={target_size}."
        )

    for record in candidates:
        if len(selected) >= target_size:
            break
        if id(record) in used_ids:
            continue
        selected.append(record)
        used_ids.add(id(record))

    if len(selected) < target_size:
        raise ValueError(
            f"Could only sample {len(selected)} rows for target_size={target_size}."
        )

    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def task_distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        label = record.get(field)
        if label is None:
            continue
        counts[str(label)] += 1
    return dict(sorted(counts.items()))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a larger article benchmark corpus from locally prepared unified sources."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/joint_reviews.article20k.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/audit/joint_reviews.article20k.audit.json"),
    )
    parser.add_argument(
        "--source-target",
        action="append",
        default=None,
        help="Per-source sample size in the form source=count.",
    )
    parser.add_argument(
        "--source-require",
        action="append",
        default=None,
        help="Per-source quota in the form source:field:value=count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_targets = parse_key_value_pairs(args.source_target or DEFAULT_SOURCE_TARGETS)
    source_requirements = parse_source_requirements(args.source_require or DEFAULT_SOURCE_REQUIRES)
    source_inputs = {
        source: DEFAULT_SOURCE_INPUTS[source]
        for source in source_targets
    }
    source_records = load_sources(source_inputs)

    merged: list[dict[str, Any]] = []
    for index, (source, target_size) in enumerate(source_targets.items()):
        sampled = sample_source_records(
            source_records[source],
            target_size=target_size,
            requirements=source_requirements.get(source, []),
            seed=args.seed + index,
        )
        merged.extend(sampled)

    random.Random(args.seed).shuffle(merged)
    write_jsonl(args.output, merged)

    split = split_unified_records(
        merged,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    audit = build_dataset_audit_report(
        merged,
        split,
        input_path=args.output,
        random_state=args.seed,
    )
    write_audit_report(args.audit_output, audit)

    test_sentiment = task_distribution(split["test"], "sentiment_label")
    test_authenticity = task_distribution(split["test"], "authenticity_label")
    print(f"wrote corpus to {args.output}")
    print(f"wrote audit to {args.audit_output}")
    print(f"records={len(merged)}")
    print(f"test_sentiment={test_sentiment}")
    print(f"test_authenticity={test_authenticity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
