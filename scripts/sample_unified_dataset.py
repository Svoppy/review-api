from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from reviewguard.data import load_unified_records


def stratified_sample(
    records: list[dict],
    *,
    max_rows: int,
    stratify_field: str | None,
    seed: int,
) -> list[dict]:
    if max_rows >= len(records):
        return list(records)

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    if not stratify_field:
        return shuffled[:max_rows]

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in shuffled:
        groups[str(record.get(stratify_field) or "__missing__")].append(record)

    sampled: list[dict] = []
    remaining = max_rows
    remaining_groups = list(groups.items())

    for index, (_, rows) in enumerate(remaining_groups, start=1):
        quota = max(1, remaining // (len(remaining_groups) - index + 1))
        take = min(quota, len(rows))
        sampled.extend(rows[:take])
        remaining -= take

    if remaining > 0:
        leftovers: list[dict] = []
        for _, rows in remaining_groups:
            leftovers.extend(rows)
        seen_ids = {id(record) for record in sampled}
        leftovers = [record for record in leftovers if id(record) not in seen_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:remaining])

    rng.shuffle(sampled)
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
        "--stratify-field",
        default=None,
        help="Optional field name for approximate stratified sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    records = load_unified_records(args.input_path)
    sampled = stratified_sample(
        records,
        max_rows=args.max_rows,
        stratify_field=args.stratify_field,
        seed=args.seed,
    )
    write_jsonl(Path(args.output_path), sampled)
    print(f"sampled {len(sampled)} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
