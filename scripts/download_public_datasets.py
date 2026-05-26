from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


HF_DATASETS = {
    "perekrestok": "lapki/perekrestok-reviews",
    "maide_up": "MichiganNLP/MAiDE-up",
}


def write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset(name: str, output_root: Path, *, max_rows: int | None) -> Path:
    dataset = load_dataset(HF_DATASETS[name])["train"]
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    rows = [dict(row) for row in dataset]
    output_path = output_root / name / f"{name}.jsonl"
    write_jsonl(output_path, rows)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public review datasets and export them as JSONL.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(HF_DATASETS),
        default=sorted(HF_DATASETS),
        help="Which public datasets to export.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw",
        help="Root directory for raw exported datasets.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap per dataset for quick smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)

    for dataset_name in args.datasets:
        output_path = export_dataset(dataset_name, output_root, max_rows=args.max_rows)
        print(f"{dataset_name}: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
