from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.data import load_unified_records
from reviewguard.training.metrics import label_field_for_task

from analyze_multiseed_statistics import model_predictions_for_task


def length_bucket(text: str) -> str:
    size = len(text.split())
    if size < 20:
        return "short"
    if size < 60:
        return "medium"
    return "long"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multitask and single-task prediction errors with slice summaries and concrete examples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=("sentiment", "authenticity"), required=True)
    parser.add_argument("--model-a", required=True, help="Model label for report output.")
    parser.add_argument("--model-a-id", required=True, help="Model id supported by analyze_multiseed_statistics helpers.")
    parser.add_argument("--model-a-export", type=Path, required=True)
    parser.add_argument("--model-b", required=True, help="Model label for report output.")
    parser.add_argument("--model-b-id", required=True, help="Model id supported by analyze_multiseed_statistics helpers.")
    parser.add_argument("--model-b-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def error_rows(
    records: list[dict[str, Any]],
    *,
    task: str,
    model_a_predictions: list[str],
    model_b_predictions: list[str],
) -> dict[str, Any]:
    label_field = label_field_for_task(task)
    slices: dict[str, Counter[str]] = defaultdict(Counter)
    a_better: list[dict[str, Any]] = []
    b_better: list[dict[str, Any]] = []

    for record, pred_a, pred_b in zip(records, model_a_predictions, model_b_predictions, strict=True):
        gold = str(record[label_field])
        meta_keys = {
            "source": str(record.get("source") or "unknown"),
            "domain": str(record.get("domain") or "unknown"),
            "language": str(record.get("language") or "unknown"),
            "length_bucket": length_bucket(str(record.get("text") or "")),
        }
        for field_name, value in meta_keys.items():
            if pred_a != gold:
                slices[f"{field_name}:{value}"]["model_a_errors"] += 1
            if pred_b != gold:
                slices[f"{field_name}:{value}"]["model_b_errors"] += 1
            slices[f"{field_name}:{value}"]["support"] += 1

        if pred_a == gold and pred_b != gold:
            a_better.append(
                {
                    "text": record["text"],
                    "gold": gold,
                    "model_a": pred_a,
                    "model_b": pred_b,
                    "source": meta_keys["source"],
                    "language": meta_keys["language"],
                    "domain": meta_keys["domain"],
                }
            )
        elif pred_b == gold and pred_a != gold:
            b_better.append(
                {
                    "text": record["text"],
                    "gold": gold,
                    "model_a": pred_a,
                    "model_b": pred_b,
                    "source": meta_keys["source"],
                    "language": meta_keys["language"],
                    "domain": meta_keys["domain"],
                }
            )

    slice_rows = []
    for key, counts in sorted(slices.items()):
        support = counts["support"]
        slice_rows.append(
            {
                "slice": key,
                "support": support,
                "model_a_error_rate": counts["model_a_errors"] / support if support else 0.0,
                "model_b_error_rate": counts["model_b_errors"] / support if support else 0.0,
                "error_gap_a_minus_b": (
                    (counts["model_a_errors"] - counts["model_b_errors"]) / support if support else 0.0
                ),
            }
        )

    return {
        "slice_rows": slice_rows,
        "model_a_better_examples": a_better,
        "model_b_better_examples": b_better,
    }


def build_markdown(
    *,
    task: str,
    model_a: str,
    model_b: str,
    payload: dict[str, Any],
    max_examples: int,
) -> str:
    lines = [
        "# Error Analysis Report",
        "",
        f"- Task: `{task}`",
        f"- Model A: `{model_a}`",
        f"- Model B: `{model_b}`",
        "",
        "## Slice-level error-rate comparison",
        "",
        "| Slice | Support | Model A error rate | Model B error rate | A-B gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["slice_rows"], key=lambda item: abs(item["error_gap_a_minus_b"]), reverse=True):
        lines.append(
            "| "
            f"`{row['slice']}` | {row['support']} | {row['model_a_error_rate']:.4f} | "
            f"{row['model_b_error_rate']:.4f} | {row['error_gap_a_minus_b']:.4f} |"
        )

    lines.extend(["", f"## Examples where `{model_a}` is correct and `{model_b}` is wrong", ""])
    for item in payload["model_a_better_examples"][:max_examples]:
        lines.append(
            f"- source=`{item['source']}` language=`{item['language']}` domain=`{item['domain']}` "
            f"gold=`{item['gold']}` a=`{item['model_a']}` b=`{item['model_b']}` text=`{item['text'][:240]}`"
        )

    lines.extend(["", f"## Examples where `{model_b}` is correct and `{model_a}` is wrong", ""])
    for item in payload["model_b_better_examples"][:max_examples]:
        lines.append(
            f"- source=`{item['source']}` language=`{item['language']}` domain=`{item['domain']}` "
            f"gold=`{item['gold']}` a=`{item['model_a']}` b=`{item['model_b']}` text=`{item['text'][:240]}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = [
        record
        for record in load_unified_records(args.input)
        if record.get(label_field_for_task(args.task)) is not None
    ]
    predictions_a = model_predictions_for_task(
        records,
        model_id=args.model_a_id,
        export_dir=args.model_a_export,
        task=args.task,
    )
    predictions_b = model_predictions_for_task(
        records,
        model_id=args.model_b_id,
        export_dir=args.model_b_export,
        task=args.task,
    )
    payload = error_rows(
        records,
        task=args.task,
        model_a_predictions=predictions_a,
        model_b_predictions=predictions_b,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_markdown(
            task=args.task,
            model_a=args.model_a,
            model_b=args.model_b,
            payload=payload,
            max_examples=args.max_examples,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote error analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
