from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.analysis.article_tables import build_article_results_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate article-facing results tables from current ReviewGuard artifacts."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--pilot-summary",
        type=Path,
        default=Path("reports/multiseed/pilot1k/summary.json"),
    )
    parser.add_argument(
        "--statistics-summary",
        type=Path,
        default=Path("reports/multiseed/pilot1k_statistics/statistics_summary.json"),
    )
    parser.add_argument(
        "--balanced-audit",
        type=Path,
        default=Path("reports/audit/joint_reviews.balanced6k.audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/article_results_package_ru.md"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("models/pilot1k-baseline/train_report.json"),
    )
    parser.add_argument(
        "--multitask-report",
        type=Path,
        default=Path("models/pilot1k-multitask/train_report.json"),
    )
    parser.add_argument(
        "--single-task-sentiment-report",
        type=Path,
        default=Path("models/pilot1k-single-task-sentiment/train_report.json"),
    )
    parser.add_argument(
        "--single-task-authenticity-report",
        type=Path,
        default=Path("models/pilot1k-single-task-authenticity/train_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = build_article_results_markdown(
        processed_dir=args.processed_dir,
        pilot_summary_path=args.pilot_summary,
        statistics_summary_path=args.statistics_summary,
        balanced_audit_path=args.balanced_audit,
        baseline_report_path=args.baseline_report,
        multitask_report_path=args.multitask_report,
        single_task_sentiment_report_path=args.single_task_sentiment_report,
        single_task_authenticity_report_path=args.single_task_authenticity_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote article tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
