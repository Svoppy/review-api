import csv
import json
from pathlib import Path

import pytest

from reviewguard.data import (
    load_maide_up,
    merge_unified_datasets,
    load_opspam,
    load_perekrestok_ratings,
    load_rureviews,
    load_unified_records,
    main,
    normalize_authenticity_label,
    normalize_sentiment_label,
    normalize_text,
    process_dataset,
    summarize_unified_records,
)


def test_normalization_helpers_cover_common_variants() -> None:
    assert normalize_text("  Отличный   товар \n\n приехал  быстро ") == "Отличный товар приехал быстро"
    assert normalize_sentiment_label("POS") == "positive"
    assert normalize_sentiment_label("3") == "neutral"
    assert normalize_authenticity_label("truthful") == "authentic"
    assert normalize_authenticity_label("ai_generated") == "fake"


def test_load_rureviews_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rureviews.csv"
    dataset_path.write_text(
        "id,text,label,product_id\n"
        "1,\" Очень хороший товар \",positive,sku-1\n",
        encoding="utf-8",
    )

    records = load_rureviews(dataset_path)

    assert len(records) == 1
    assert records[0].source == "rureviews"
    assert records[0].language == "ru"
    assert records[0].sentiment_label == "positive"
    assert records[0].text == "Очень хороший товар"


def test_load_rureviews_tab_separated_export_uses_last_field_as_label(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rureviews_export.csv"
    dataset_path.write_text(
        "review\tsentiment\n"
        "Ткань ужасная\tрукав короткий\tnegative\n",
        encoding="utf-8",
    )

    records = load_rureviews(dataset_path)

    assert len(records) == 1
    assert records[0].sentiment_label == "negative"
    assert records[0].text == "Ткань ужасная рукав короткий"


def test_load_rureviews_tab_separated_export_supports_multiline_reviews(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rureviews_multiline.csv"
    dataset_path.write_text(
        "review\tsentiment\n"
        "\"Первая строка отзыва\n"
        "и продолжение во второй строке\"\tpositive\n",
        encoding="utf-8",
    )

    records = load_rureviews(dataset_path)

    assert len(records) == 1
    assert records[0].sentiment_label == "positive"
    assert "Первая строка отзыва и продолжение во второй строке" == records[0].text


def test_load_perekrestok_from_jsonl_derives_sentiment(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perekrestok.jsonl"
    dataset_path.write_text(
        json.dumps({"review_id": "7", "comment": "Нормально", "rating": 3, "sku": "milk-1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = load_perekrestok_ratings(dataset_path)

    assert len(records) == 1
    assert records[0].sentiment_label == "neutral"
    assert records[0].rating == 3.0
    assert records[0].product_id == "milk-1"


def test_load_opspam_from_directory_infers_labels(tmp_path: Path) -> None:
    sample = tmp_path / "opspam" / "deceptive" / "positive" / "fold1"
    sample.mkdir(parents=True)
    review_path = sample / "d1.txt"
    review_path.write_text("Amazing hotel and perfect service.", encoding="utf-8")

    records = load_opspam(tmp_path / "opspam")

    assert len(records) == 1
    assert records[0].authenticity_label == "fake"
    assert records[0].sentiment_label == "positive"
    assert records[0].domain == "hospitality"
    assert records[0].metadata["path"] == "deceptive/positive/fold1/d1.txt"


def test_load_maide_up_supports_boolean_ai_flag(tmp_path: Path) -> None:
    dataset_path = tmp_path / "maide.json"
    dataset_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "id": "m1",
                        "text": "This blender works fine.",
                        "is_ai_generated": True,
                        "rating": "4",
                        "language": "en",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_maide_up(dataset_path)

    assert len(records) == 1
    assert records[0].authenticity_label == "fake"
    assert records[0].sentiment_label == "positive"
    assert records[0].source == "maide_up"


def test_load_maide_up_supports_huggingface_shape(tmp_path: Path) -> None:
    dataset_path = tmp_path / "maide_hf.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "Unnamed: 0": 7,
                "Review_Language": "Chinese",
                "City Name": "Ankara",
                "Hotel Name": "Sample Hotel",
                "Upside_Review": "Beautiful lobby and kind staff.",
                "Downside_Review": "Air conditioning was weak.",
                "Review_Score": 4.0,
                "Sentiment": "NEG",
                "source": 1,
                "Prompt_Language": "English",
                "na_up_review": False,
                "na_down_review": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_maide_up(dataset_path)

    assert len(records) == 1
    assert records[0].language == "chinese"
    assert records[0].domain == "hospitality"
    assert records[0].authenticity_label == "fake"
    assert records[0].sentiment_label == "negative"
    assert records[0].record_id == "7:1:Chinese:Sample Hotel"
    assert records[0].title == "Sample Hotel"
    assert records[0].product_id == "Sample Hotel"
    assert records[0].metadata["city_name"] == "Ankara"
    assert "Pros:" in records[0].text
    assert "Cons:" in records[0].text


def test_process_dataset_writes_jsonl_and_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "ru.csv"
    input_path.write_text("text,label\nХорошо,positive\n", encoding="utf-8")

    jsonl_path = tmp_path / "processed.jsonl"
    csv_path = tmp_path / "processed.csv"

    process_dataset(dataset="rureviews", input_path=input_path, output_path=jsonl_path)
    process_dataset(dataset="rureviews", input_path=input_path, output_path=csv_path)

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["source"] == "rureviews"

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["sentiment_label"] == "positive"


def test_load_unified_records_restores_processed_metadata(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.jsonl"
    processed_path.write_text(
        json.dumps(
            {
                "record_id": "1",
                "source": "rureviews",
                "language": "ru",
                "domain": "ecommerce",
                "sentiment_label": "positive",
                "authenticity_label": None,
                "text": "Отличный товар",
                "metadata": {"raw_dataset": "RuReviews"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_unified_records(processed_path)

    assert len(records) == 1
    assert records[0]["source"] == "rureviews"
    assert records[0]["metadata"]["raw_dataset"] == "RuReviews"


def test_cli_entrypoint_processes_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("text,label\nПлохо,negative\n", encoding="utf-8")

    exit_code = main(
        [
            "normalize",
            "--dataset",
            "rureviews",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["sentiment_label"] == "negative"


def test_merge_unified_datasets_deduplicates_and_summarizes(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "ru-1",
                        "source": "rureviews",
                        "language": "ru",
                        "domain": "ecommerce",
                        "text": "Отличный товар",
                        "sentiment_label": "positive",
                        "authenticity_label": None,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source": "opspam",
                        "language": "en",
                        "domain": "hospitality",
                        "text": "Fake review text",
                        "sentiment_label": "negative",
                        "authenticity_label": "fake",
                        "metadata": {},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "ru-1",
                        "source": "rureviews",
                        "language": "ru",
                        "domain": "ecommerce",
                        "text": "Отличный товар",
                        "sentiment_label": "positive",
                        "authenticity_label": None,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source": "maide_up",
                        "language": "en",
                        "domain": "ecommerce",
                        "text": "AI review",
                        "sentiment_label": None,
                        "authenticity_label": "fake",
                        "metadata": {},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = merge_unified_datasets([first, second])
    summary = summarize_unified_records(records)

    assert len(records) == 3
    assert summary["records"] == 3
    assert summary["sentiment_labeled"] == 2
    assert summary["authenticity_labeled"] == 2
    assert summary["sources"]["rureviews"] == 1


def test_merge_unified_datasets_keeps_distinct_same_text_records_without_ids(tmp_path: Path) -> None:
    first = tmp_path / "same_text_a.jsonl"
    second = tmp_path / "same_text_b.jsonl"
    payload = {
        "source": "rureviews",
        "language": "ru",
        "domain": "ecommerce",
        "text": "Хорошо",
        "sentiment_label": "positive",
        "authenticity_label": None,
        "metadata": {},
    }
    first.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    second.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    records = merge_unified_datasets([first, second])

    assert len(records) == 2


def test_merge_unified_datasets_merges_same_record_id_with_complementary_labels(tmp_path: Path) -> None:
    first = tmp_path / "with_sentiment.jsonl"
    second = tmp_path / "with_authenticity.jsonl"
    first.write_text(
        json.dumps(
            {
                "record_id": "42",
                "source": "shared",
                "language": "en",
                "domain": "ecommerce",
                "text": "Great value",
                "sentiment_label": "positive",
                "authenticity_label": None,
                "metadata": {"left": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "record_id": "42",
                "source": "shared",
                "language": "en",
                "domain": "ecommerce",
                "text": "Great value",
                "sentiment_label": None,
                "authenticity_label": "authentic",
                "metadata": {"right": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = merge_unified_datasets([first, second])

    assert len(records) == 1
    assert records[0]["sentiment_label"] == "positive"
    assert records[0]["authenticity_label"] == "authentic"
    assert records[0]["metadata"] == {"left": True, "right": True}


def test_merge_unified_datasets_rejects_conflicting_duplicate_labels(tmp_path: Path) -> None:
    first = tmp_path / "conflict_a.jsonl"
    second = tmp_path / "conflict_b.jsonl"
    base = {
        "record_id": "42",
        "source": "shared",
        "language": "en",
        "domain": "ecommerce",
        "text": "Great value",
        "authenticity_label": None,
        "metadata": {},
    }
    first.write_text(
        json.dumps({**base, "sentiment_label": "positive"}) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({**base, "sentiment_label": "negative"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting sentiment_label values"):
        merge_unified_datasets([first, second])


def test_merge_unified_datasets_rejects_same_id_with_mismatched_text(tmp_path: Path) -> None:
    first = tmp_path / "mismatch_a.jsonl"
    second = tmp_path / "mismatch_b.jsonl"
    first.write_text(
        json.dumps(
            {
                "record_id": "42",
                "source": "shared",
                "language": "en",
                "domain": "ecommerce",
                "text": "Great value",
                "sentiment_label": "positive",
                "authenticity_label": None,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "record_id": "42",
                "source": "shared",
                "language": "en",
                "domain": "ecommerce",
                "text": "Completely different review",
                "sentiment_label": None,
                "authenticity_label": "authentic",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting text values"):
        merge_unified_datasets([first, second])


def test_data_cli_no_args_prints_top_level_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "normalize" in captured.out
    assert "merge" in captured.out


def test_data_cli_help_keeps_top_level_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "normalize" in captured.out
    assert "merge" in captured.out
