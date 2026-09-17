from __future__ import annotations

import argparse
from pathlib import Path

from sample_unified_dataset import stratified_sample, write_jsonl
from reviewguard.data import load_unified_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a family of deterministic corpora for learning-curve experiments."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-fields",
        nargs="+",
        default=["source", "sentiment_label", "authenticity_label"],
    )
    parser.add_argument("--min-per-group", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_unified_records(args.input)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        sampled = stratified_sample(
            records,
            max_rows=size,
            stratify_fields=list(args.stratify_fields),
            seed=args.seed + size,
            min_per_group=args.min_per_group,
        )
        output_path = args.output_root / f"{args.input.stem}.{size}.jsonl"
        write_jsonl(output_path, sampled)
        print(f"{size}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
