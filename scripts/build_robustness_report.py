from __future__ import annotations

import argparse
import json
from pathlib import Path

from reviewguard.analysis.robustness import build_multitask_robustness_report
from reviewguard.data import load_unified_records
from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS


def _load_predictions(path: Path) -> dict[str, list[str] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sentiment": payload.get("sentiment"),
        "authenticity": payload.get("authenticity"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build slice-based robustness diagnostics from aligned predictions."
    )
    parser.add_argument("--input", required=True, type=Path, help="Unified JSONL evaluation corpus.")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="JSON file with aligned sentiment/authenticity prediction arrays.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum labeled rows required to report a slice.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_unified_records(args.input)
    predictions = _load_predictions(args.predictions)
    report = build_multitask_robustness_report(
        records,
        sentiment_predictions=predictions["sentiment"],
        authenticity_predictions=predictions["authenticity"],
        sentiment_labels=SENTIMENT_LABELS,
        authenticity_labels=AUTHENTICITY_LABELS,
        min_support=args.min_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote robustness report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
