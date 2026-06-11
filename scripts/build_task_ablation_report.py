from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.analysis.ablation import build_ablation_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an article-ready single-task vs multitask ablation summary "
            "from the existing multi-seed and statistical reports."
        )
    )
    parser.add_argument(
        "--multiseed-summary",
        type=Path,
        default=Path("reports/multiseed/pilot1k/summary.json"),
    )
    parser.add_argument(
        "--statistics-summary",
        type=Path,
        default=Path("reports/multiseed/pilot1k_statistics/statistics_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/multiseed/pilot1k_ablation"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def task_heading(task: str) -> str:
    return "Sentiment" if task == "sentiment" else "Authenticity"


def build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Single-Task vs Multitask Ablation",
        "",
        "This report isolates the core architectural question of the pilot: how the shared multitask encoder behaves relative to the task-specific Transformer baselines and the classical baseline on each task.",
        "",
        "## Summary Table",
        "",
        "| Task | Baseline Macro-F1 | Single-task Macro-F1 | Multitask Macro-F1 | Multitask vs Single-task | Multitask vs Baseline |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{task_heading(row['task'])}` | "
            f"{row['baseline_macro_f1_mean']:.4f} +/- {row['baseline_macro_f1_std']:.4f} | "
            f"{row['single_task_macro_f1_mean']:.4f} +/- {row['single_task_macro_f1_std']:.4f} | "
            f"{row['multitask_macro_f1_mean']:.4f} +/- {row['multitask_macro_f1_std']:.4f} | "
            f"{row['multitask_vs_single_task_note']} | "
            f"{row['multitask_vs_baseline_note']} |"
        )

    lines.extend(["", "## Interpretation", ""])
    for row in rows:
        task = task_heading(row["task"])
        lines.append(f"### {task}")
        lines.append("")
        lines.append(
            "- "
            + (
                f"Baseline macro-F1: `{row['baseline_macro_f1_mean']:.4f} +/- {row['baseline_macro_f1_std']:.4f}`; "
                f"single-task macro-F1: `{row['single_task_macro_f1_mean']:.4f} +/- {row['single_task_macro_f1_std']:.4f}`; "
                f"multitask macro-F1: `{row['multitask_macro_f1_mean']:.4f} +/- {row['multitask_macro_f1_std']:.4f}`."
            )
        )
        lines.append(
            "- "
            + (
                f"Multitask vs single-task: `{row['multitask_vs_single_task_delta']:+.4f}` macro-F1; "
                f"{row['multitask_vs_single_task_note']}."
            )
        )
        lines.append(
            "- "
            + (
                f"Multitask vs baseline: `{row['multitask_vs_baseline_delta']:+.4f}` macro-F1; "
                f"{row['multitask_vs_baseline_note']}."
            )
        )
        if row["task"] == "authenticity":
            lines.append(
                "- The current pilot supports a positive-transfer interpretation for authenticity: the shared encoder improves the trust-oriented task relative to both comparison families."
            )
        else:
            lines.append(
                "- The current pilot does not support a positive-transfer interpretation for sentiment: multitask learning is roughly tied with the single-task Transformer on macro-F1 but remains clearly below the classical baseline."
            )
        lines.append("")

    lines.extend(
        [
            "## Article-ready takeaway",
            "",
            "- The pilot ablation supports asymmetric transfer rather than uniform multitask gains.",
            "- For authenticity, shared representations appear beneficial on the current low-resource mixed corpus.",
            "- For sentiment, shared training does not beat the single-task Transformer and still underperforms the baseline, which means the current pilot cannot claim broad multitask superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    multiseed_summary = json.loads(args.multiseed_summary.read_text(encoding="utf-8"))
    statistics_summary = json.loads(args.statistics_summary.read_text(encoding="utf-8"))

    rows = build_ablation_rows(
        multiseed_summary=multiseed_summary,
        pairwise_rows=[
            {
                "task": item["task"],
                "metric": metric_name,
                "model_a": item["model_a"],
                "model_b": item["model_b"],
                "observed_delta": metric_payload["observed_delta"],
                "ci_low": metric_payload["bootstrap_ci95"]["ci_low"],
                "ci_high": metric_payload["bootstrap_ci95"]["ci_high"],
                "p_value": metric_payload["approx_randomization_p_value"],
            }
            for item in statistics_summary["pairwise_comparisons"]
            for metric_name, metric_payload in item["metrics"].items()
        ],
    )

    payload = {
        "rows": rows,
        "source_multiseed_summary": str(args.multiseed_summary),
        "source_statistics_summary": str(args.statistics_summary),
    }
    (args.output_dir / "task_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_ablation.csv", rows)
    (args.output_dir / "task_ablation.md").write_text(
        build_report(rows) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote ablation JSON to {args.output_dir / 'task_ablation.json'}")
    print(f"Wrote ablation CSV to {args.output_dir / 'task_ablation.csv'}")
    print(f"Wrote ablation report to {args.output_dir / 'task_ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
