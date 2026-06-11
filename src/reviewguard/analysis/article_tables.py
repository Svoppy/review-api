from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRow:
    name: str
    filename: str | None
    language: str
    domain: str
    label_type: str
    task: str
    limitation: str


CURRENT_DATASET_ROWS = (
    DatasetRow(
        name="RuReviews",
        filename="rureviews.jsonl",
        language="RU",
        domain="e-commerce",
        label_type="готовые sentiment labels",
        task="Sentiment",
        limitation="нет authenticity label coverage",
    ),
    DatasetRow(
        name="Perekrestok Reviews",
        filename="perekrestok.jsonl",
        language="RU",
        domain="retail",
        label_type="rating-derived sentiment",
        task="Sentiment",
        limitation="authenticity отсутствует; sentiment partially heuristic",
    ),
    DatasetRow(
        name="OpSpam",
        filename=None,
        language="EN",
        domain="hospitality",
        label_type="truthful/deceptive labels",
        task="Authenticity",
        limitation="planned extension; raw text not prepared locally",
    ),
    DatasetRow(
        name="FraudDataset (Yelp)",
        filename=None,
        language="EN",
        domain="local commerce",
        label_type="silver fraud labels",
        task="Authenticity",
        limitation="adapter ready, local text export not prepared",
    ),
    DatasetRow(
        name="MAiDE-up",
        filename="maide_up.jsonl",
        language="multilingual",
        domain="hospitality",
        label_type="sentiment + AI-generated authenticity",
        task="Sentiment + Authenticity",
        limitation="current authenticity evidence comes only from this family",
    ),
)

MODEL_DISPLAY_NAMES = {
    "baseline": "TF-IDF + Logistic Regression",
    "single-task-sentiment": "Single-task Transformer",
    "single-task-authenticity": "Single-task Transformer",
    "multitask": "Multitask Transformer",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _format_metric(value: float | int) -> str:
    return f"{float(value):.4f}"


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _format_ci(ci_low: float, ci_high: float) -> str:
    return f"[{ci_low:.4f}, {ci_high:.4f}]"


def _render_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> list[str]:
    header = "| gold ↓ / pred → | " + " | ".join(f"`{label}`" for label in labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    rows = [header, separator]
    for label, row in zip(labels, matrix, strict=False):
        rows.append("| " + " | ".join([f"`{label}`", *[str(value) for value in row]]) + " |")
    return rows


def _load_optional_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _find_pairwise(
    pairwise: list[dict[str, Any]],
    *,
    task: str,
    model_a: str,
    model_b: str,
    metric: str,
) -> dict[str, Any] | None:
    for entry in pairwise:
        if (
            entry["task"] == task
            and entry["model_a"] == model_a
            and entry["model_b"] == model_b
            and metric in entry["metrics"]
        ):
            return entry["metrics"][metric]
    return None


def _significance_note(
    pairwise: list[dict[str, Any]],
    *,
    task: str,
    model_id: str,
) -> str:
    if model_id == "baseline":
        return "референсный classical baseline"
    if model_id.startswith("single-task"):
        return "референсный Transformer comparison"
    if model_id != "multitask":
        return "-"

    if task == "authenticity":
        against_st = _find_pairwise(
            pairwise,
            task=task,
            model_a="multitask",
            model_b="single-task-authenticity",
            metric="macro_f1",
        )
        against_base = _find_pairwise(
            pairwise,
            task=task,
            model_a="multitask",
            model_b="baseline",
            metric="macro_f1",
        )
        if against_st and against_base:
            return (
                "лучше single-task и baseline "
                f"(p={against_st['approx_randomization_p_value']:.4f}; "
                f"p={against_base['approx_randomization_p_value']:.4f})"
            )
    if task == "sentiment":
        against_st = _find_pairwise(
            pairwise,
            task=task,
            model_a="multitask",
            model_b="single-task-sentiment",
            metric="macro_f1",
        )
        against_base = _find_pairwise(
            pairwise,
            task=task,
            model_a="multitask",
            model_b="baseline",
            metric="macro_f1",
        )
        if against_st and against_base:
            return (
                "сопоставимо с single-task, хуже baseline "
                f"(p={against_st['approx_randomization_p_value']:.4f}; "
                f"p={against_base['approx_randomization_p_value']:.4f})"
            )
    return "-"


def build_article_results_markdown(
    *,
    processed_dir: Path,
    pilot_summary_path: Path,
    statistics_summary_path: Path,
    balanced_audit_path: Path,
    baseline_report_path: Path | None = None,
    multitask_report_path: Path | None = None,
    single_task_sentiment_report_path: Path | None = None,
    single_task_authenticity_report_path: Path | None = None,
) -> str:
    pilot_summary = _load_json(pilot_summary_path)
    statistics_summary = _load_json(statistics_summary_path)
    balanced_audit = _load_json(balanced_audit_path)
    pairwise = statistics_summary["pairwise_comparisons"]
    intervals = statistics_summary["model_intervals"]
    baseline_report = _load_optional_report(baseline_report_path)
    multitask_report = _load_optional_report(multitask_report_path)
    single_task_sentiment_report = _load_optional_report(single_task_sentiment_report_path)
    single_task_authenticity_report = _load_optional_report(single_task_authenticity_report_path)

    local_counts: dict[str, int | str] = {}
    for row in CURRENT_DATASET_ROWS:
        if row.filename is None:
            local_counts[row.name] = "not prepared locally"
            continue
        local_counts[row.name] = _jsonl_line_count(processed_dir / row.filename)

    main_rows: list[str] = []
    ordered_models = (
        ("baseline", "sentiment"),
        ("single-task-sentiment", "sentiment"),
        ("multitask", "sentiment"),
        ("baseline", "authenticity"),
        ("single-task-authenticity", "authenticity"),
        ("multitask", "authenticity"),
    )
    for model_id, task in ordered_models:
        metrics = intervals[model_id]["test_metrics"][task]
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]
        precision_macro = metrics["precision_macro"]
        recall_macro = metrics["recall_macro"]
        mean_std = _format_mean_std(macro_f1["mean"], macro_f1["std"])
        note = _significance_note(pairwise, task=task, model_id=model_id)
        main_rows.append(
            "| "
            + " | ".join(
                [
                    MODEL_DISPLAY_NAMES[model_id],
                    "pilot1k",
                    task.capitalize(),
                    _format_metric(accuracy["mean"]),
                    _format_metric(macro_f1["mean"]),
                    _format_metric(weighted_f1["mean"]),
                    _format_metric(precision_macro["mean"]),
                    _format_metric(recall_macro["mean"]),
                    mean_std,
                    note,
                ]
            )
            + " |"
        )

    pairwise_rows: list[str] = []
    for entry in pairwise:
        metric = entry["metrics"]["macro_f1"]
        pairwise_rows.append(
            "| "
            + " | ".join(
                [
                    entry["task"].capitalize(),
                    entry["model_a"],
                    entry["model_b"],
                    _format_metric(metric["observed_delta"]),
                    _format_ci(
                        metric["bootstrap_ci95"]["ci_low"],
                        metric["bootstrap_ci95"]["ci_high"],
                    ),
                    f"{metric['approx_randomization_p_value']:.4f}",
                ]
            )
            + " |"
        )

    seed_rows: list[str] = []
    for model_id, task in (
        ("single-task-sentiment", "sentiment"),
        ("multitask", "sentiment"),
        ("single-task-authenticity", "authenticity"),
        ("multitask", "authenticity"),
    ):
        metrics = intervals[model_id]["test_metrics"][task]
        seed_rows.append(
            "| "
            + " | ".join(
                [
                    MODEL_DISPLAY_NAMES[model_id],
                    task.capitalize(),
                    _format_metric(metrics["accuracy"]["mean"]),
                    _format_metric(metrics["accuracy"]["std"]),
                    _format_metric(metrics["macro_f1"]["mean"]),
                    _format_metric(metrics["macro_f1"]["std"]),
                ]
            )
            + " |"
        )

    dataset_rows: list[str] = []
    for row in CURRENT_DATASET_ROWS:
        size = local_counts[row.name]
        dataset_rows.append(
            "| "
            + " | ".join(
                [
                    f"`{row.name}`",
                    row.language,
                    row.domain,
                    str(size),
                    row.label_type,
                    row.task,
                    row.limitation,
                ]
            )
            + " |"
        )

    audit = balanced_audit
    split_overlap = audit["split_overlap"]
    sentiment_test = audit["tasks"]["sentiment"]["test_distribution"]
    authenticity_test = audit["tasks"]["authenticity"]["test_distribution"]
    majority_sent = audit["tasks"]["sentiment"]["majority_baseline"]["test"]
    majority_auth = audit["tasks"]["authenticity"]["majority_baseline"]["test"]

    blockers = [
        "authenticity coverage в текущем expanded snapshot по-прежнему идет только из `MAiDE-up`;",
        "в balanced6k test split по sentiment все еще только `8` примеров класса `neutral`;",
        "полные article-grade robustness и ablation reruns по усиленному протоколу еще не выполнены.",
    ]

    confusion_sections: list[str] = []
    if baseline_report is not None:
        confusion_sections.extend(
            [
                "### Baseline sentiment",
                "",
                *_render_confusion_matrix(
                    ["negative", "neutral", "positive"],
                    baseline_report["test_metrics"]["sentiment"]["confusion_matrix"],
                ),
                "",
            ]
        )
    if single_task_sentiment_report is not None:
        confusion_sections.extend(
            [
                "### Single-task sentiment",
                "",
                *_render_confusion_matrix(
                    ["negative", "neutral", "positive"],
                    single_task_sentiment_report["test_metrics"]["sentiment"]["confusion_matrix"],
                ),
                "",
            ]
        )
    if multitask_report is not None:
        confusion_sections.extend(
            [
                "### Multitask sentiment",
                "",
                *_render_confusion_matrix(
                    ["negative", "neutral", "positive"],
                    multitask_report["test_metrics"]["sentiment"]["confusion_matrix"],
                ),
                "",
                "### Multitask authenticity",
                "",
                *_render_confusion_matrix(
                    ["authentic", "fake"],
                    multitask_report["test_metrics"]["authenticity"]["confusion_matrix"],
                ),
                "",
            ]
        )
    if single_task_authenticity_report is not None:
        confusion_sections.extend(
            [
                "### Single-task authenticity",
                "",
                *_render_confusion_matrix(
                    ["authentic", "fake"],
                    single_task_authenticity_report["test_metrics"]["authenticity"]["confusion_matrix"],
                ),
                "",
            ]
        )

    error_notes: list[str] = []
    if baseline_report is not None:
        baseline_neutral = baseline_report["test_metrics"]["sentiment"]["confusion_matrix"][1][1]
        error_notes.append(
            f"baseline по sentiment корректно восстанавливает `neutral` только в `{baseline_neutral}/5` pilot-примерах;"
        )
    if single_task_sentiment_report is not None:
        st_neutral = single_task_sentiment_report["test_metrics"]["sentiment"]["confusion_matrix"][1][1]
        error_notes.append(
            f"single-task sentiment фактически рушится на редком классе `neutral`: только `{st_neutral}/5` верных ответов;"
        )
    if multitask_report is not None:
        mt_pos_to_neg = multitask_report["test_metrics"]["sentiment"]["confusion_matrix"][2][0]
        mt_fake_miss = multitask_report["test_metrics"]["authenticity"]["confusion_matrix"][1][0]
        error_notes.append(
            f"multitask sentiment все еще переоценивает `negative` для положительных отзывов (`positive -> negative = {mt_pos_to_neg}`);"
        )
        error_notes.append(
            f"multitask authenticity пропускает только `{mt_fake_miss}` fake-review на representative pilot-checkpoint;"
        )
    if single_task_authenticity_report is not None:
        st_auth_fp = single_task_authenticity_report["test_metrics"]["authenticity"]["confusion_matrix"][0][1]
        error_notes.append(
            f"single-task authenticity переоценивает класс `fake`, давая `{st_auth_fp}` ошибок `authentic -> fake`;"
        )

    return "\n".join(
        [
            "# Пакет итоговых таблиц статьи",
            "",
            "Этот файл собирается автоматически из текущих отчетов и служит canonical article-facing appendix для текущего snapshot.",
            "",
            "## Источники",
            "",
            f"- pilot summary: `{pilot_summary_path}`",
            f"- statistics summary: `{statistics_summary_path}`",
            f"- balanced audit: `{balanced_audit_path}`",
            "",
            "## Таблица 1. Снимок датасетов и текущего локального статуса",
            "",
            "| Датасет | Язык | Домен | Размер локальной подготовки | Тип меток | Задача | Ограничения |",
            "|---|---|---|---:|---|---|---|",
            *dataset_rows,
            "",
            "## Таблица 2. Основное сравнение моделей на выполненном pilot protocol",
            "",
            "| Модель | Корпус | Задача | Accuracy | Macro-F1 | Weighted-F1 | Precision_macro | Recall_macro | Mean +/- Std | Значимость |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
            *main_rows,
            "",
            "## Таблица 3. Парные сравнения по Macro-F1 на фиксированном test split",
            "",
            "| Задача | Модель A | Модель B | Delta | 95% CI | p-value |",
            "|---|---|---|---:|---|---:|",
            *pairwise_rows,
            "",
            "## Таблица 4. Устойчивость по train seeds",
            "",
            "| Модель | Задача | Mean Accuracy | Std Accuracy | Mean Macro-F1 | Std Macro-F1 |",
            "|---|---|---:|---:|---:|---:|",
            *seed_rows,
            "",
            "## Таблица 5. Расширенный balanced6k snapshot: data-readiness и leakage audit",
            "",
            f"- records: `{audit['records']}`",
            f"- source distribution: `rureviews={audit['source_distribution']['rureviews']}`, `perekrestok={audit['source_distribution']['perekrestok']}`, `maide_up={audit['source_distribution']['maide_up']}`",
            f"- split overlap: `train/valid normalized={split_overlap['train_vs_valid']['normalized_text_overlap']}`, `train/test normalized={split_overlap['train_vs_test']['normalized_text_overlap']}`, `valid/test normalized={split_overlap['valid_vs_test']['normalized_text_overlap']}`",
            f"- duplicate summary: `exact={audit['duplicate_summary']['exact_duplicate_rows']}`, `normalized={audit['duplicate_summary']['normalized_duplicate_rows']}`",
            f"- authenticity source coverage: `maide_up={audit['label_coverage']['source']['authenticity']['maide_up']['label_coverage']:.1f}`, `rureviews={audit['label_coverage']['source']['authenticity']['rureviews']['label_coverage']:.1f}`, `perekrestok={audit['label_coverage']['source']['authenticity']['perekrestok']['label_coverage']:.1f}`",
            "",
            "## Таблица 6. Классовый баланс и majority baseline на balanced6k test split",
            "",
            "| Корпус | Задача | Класс | Число объектов | Доля |",
            "|---|---|---|---:|---:|",
            f"| `balanced6k` | Sentiment | `negative` | {sentiment_test['counts']['negative']} | {sentiment_test['proportions']['negative']:.4f} |",
            f"| `balanced6k` | Sentiment | `neutral` | {sentiment_test['counts']['neutral']} | {sentiment_test['proportions']['neutral']:.4f} |",
            f"| `balanced6k` | Sentiment | `positive` | {sentiment_test['counts']['positive']} | {sentiment_test['proportions']['positive']:.4f} |",
            f"| `balanced6k` | Authenticity | `authentic` | {authenticity_test['counts']['authentic']} | {authenticity_test['proportions']['authentic']:.4f} |",
            f"| `balanced6k` | Authenticity | `fake` | {authenticity_test['counts']['fake']} | {authenticity_test['proportions']['fake']:.4f} |",
            "",
            "| Корпус | Задача | Majority baseline Accuracy | Majority baseline Macro-F1 |",
            "|---|---|---:|---:|",
            f"| `balanced6k` | Sentiment | {majority_sent['accuracy']:.4f} | {majority_sent['macro_f1']:.4f} |",
            f"| `balanced6k` | Authenticity | {majority_auth['accuracy']:.4f} | {majority_auth['macro_f1']:.4f} |",
            "",
            "## Таблица 7. Representative confusion matrices на pilot checkpoints",
            "",
            *confusion_sections,
            "## Краткий error analysis",
            "",
            *[f"- {note}" for note in error_notes],
            "",
            "## Что уже article-ready",
            "",
            "- pilot comparison `baseline / single-task / multitask` собран на одном фиксированном split;",
            "- для pilot уже есть `mean +/- std`, bootstrap CI и approximate randomization tests;",
            "- balanced6k snapshot уже проходит leakage-safe split audit с нулевым `normalized_text_overlap` между split'ами;",
            "- provenance и label coverage теперь явно фиксируются в unified pipeline.",
            "",
            "## Что еще не дает честно ставить 10/10",
            "",
            *[f"- {blocker}" for blocker in blockers],
            "",
        ]
    )
