import csv
import json
from pathlib import Path

from reviewguard.data import (
    load_maide_up,
    load_opspam,
    load_perekrestok_ratings,
    load_rureviews,
    load_unified_records,
    main,
    normalize_authenticity_label,
    normalize_sentiment_label,
    normalize_text,
    process_dataset,
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
