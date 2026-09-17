from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


HF_DATASETS = {
    "perekrestok": "lapki/perekrestok-reviews",
    "maide_up": "MichiganNLP/MAiDE-up",
    "wildberries": "Hplss/wb-review-dataset",
}

DATASET_PROVENANCE = {
    "perekrestok": {
        "dataset_card": "https://huggingface.co/datasets/lapki/perekrestok-reviews",
        "license": "Refer to the upstream dataset card before redistribution.",
    },
    "maide_up": {
        "dataset_card": "https://huggingface.co/datasets/MichiganNLP/MAiDE-up",
        "license": "Refer to the upstream dataset card before redistribution.",
    },
    "wildberries": {
        "dataset_card": "https://huggingface.co/datasets/Hplss/wb-review-dataset",
        "license": "CC BY-NC-SA 4.0",
        "intended_use": "Research and non-commercial ML only; attribution and share-alike apply.",
        "privacy_note": "The upstream card states that review and product IDs are pseudonymous and phone numbers/emails are redacted.",
    },
}


def write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(name: str, output_path: Path, rows: int) -> Path:
    provenance_path = output_path.with_name(f"{name}.provenance.json")
    payload = {
        "dataset_name": name,
        "dataset_id": HF_DATASETS[name],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_export": {
            "path": str(output_path),
            "rows": rows,
            "sha256": sha256(output_path),
        },
        **DATASET_PROVENANCE[name],
    }
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance_path


def export_dataset(name: str, output_root: Path, *, max_rows: int | None) -> Path:
    dataset = load_dataset(HF_DATASETS[name])["train"]
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    rows = [dict(row) for row in dataset]
    output_path = output_root / name / f"{name}.jsonl"
    write_jsonl(output_path, rows)
    write_provenance(name, output_path, len(rows))
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
