from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from reviewguard.data import load_unified_records, normalize_text


def normalize_source_name(value: str) -> str:
    normalized = normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "perekrestok_reviews": "perekrestok",
        "perekrestok_review": "perekrestok",
        "ru_reviews": "rureviews",
    }
    return aliases.get(normalized, normalized)


def _group_value(record: dict, fields: list[str]) -> str:
    return "|".join(f"{field}={record.get(field) or '__missing__'}" for field in fields)


def _parse_source_caps(values: list[str] | None) -> dict[str, int]:
    caps: dict[str, int] = {}
    for value in values or []:
        name, raw_limit = value.split("=", 1)
        caps[normalize_source_name(name)] = int(raw_limit)
    return caps


def stratified_sample(
    records: list[dict],
    *,
    max_rows: int,
    stratify_fields: list[str],
    seed: int,
    min_per_group: int,
    source_caps: dict[str, int] | None = None,
) -> list[dict]:
    if max_rows >= len(records):
        candidates = list(records)
    else:
        rng = random.Random(seed)
        candidates = list(records)
        rng.shuffle(candidates)

    if source_caps:
        source_counts: dict[str, int] = defaultdict(int)
        filtered: list[dict] = []
        for record in candidates:
            source = str(record.get("source") or "__missing__")
            cap = source_caps.get(source)
            if cap is not None and source_counts[source] >= cap:
                continue
            filtered.append(record)
            source_counts[source] += 1
        candidates = filtered

    if not stratify_fields:
        return candidates[:max_rows]

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        groups[_group_value(record, stratify_fields)].append(record)

    sampled: list[dict] = []
    used_ids: set[int] = set()
    for rows in groups.values():
        take = min(len(rows), min_per_group)
        sampled.extend(rows[:take])
        used_ids.update(id(record) for record in rows[:take])

    if len(sampled) >= max_rows:
        return sampled[:max_rows]

    remaining = max_rows
    remaining -= len(sampled)
    remaining_groups = list(groups.items())

    for index, (_, rows) in enumerate(remaining_groups, start=1):
        rows = [record for record in rows if id(record) not in used_ids]
        if not rows:
            continue
        quota = max(1, remaining // (len(remaining_groups) - index + 1))
        take = min(quota, len(rows))
        sampled.extend(rows[:take])
        used_ids.update(id(record) for record in rows[:take])
        remaining -= take

    if remaining > 0:
        leftovers: list[dict] = []
        for _, rows in remaining_groups:
            leftovers.extend(rows)
        leftovers = [record for record in leftovers if id(record) not in used_ids]
        random.Random(seed).shuffle(leftovers)
        sampled.extend(leftovers[:remaining])

    random.Random(seed).shuffle(sampled)
    return sampled[:max_rows]


def write_jsonl(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic sample from a unified review corpus.")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--max-rows", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-fields",
        nargs="+",
        default=[],
        help="Optional field names for approximate composite stratified sampling.",
    )
    parser.add_argument(
        "--min-per-group",
        type=int,
        default=0,
        help="Reserve at least this many rows for each stratum before filling the remaining budget.",
    )
    parser.add_argument(
        "--source-cap",
        action="append",
        default=[],
        help="Optional source cap in the form source_name=count. Repeat the flag for multiple sources.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    records = load_unified_records(args.input_path)
    sampled = stratified_sample(
        records,
        max_rows=args.max_rows,
        stratify_fields=list(args.stratify_fields),
        seed=args.seed,
        min_per_group=args.min_per_group,
        source_caps=_parse_source_caps(args.source_cap),
    )
    write_jsonl(Path(args.output_path), sampled)
    print(f"sampled {len(sampled)} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
