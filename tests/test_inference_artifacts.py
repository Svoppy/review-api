import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

sys.path.insert(0, "src")

from reviewguard.ml.inference import ReviewAnalyzer


def test_analyzer_readiness_requires_export_files(tmp_path: Path) -> None:
    analyzer = ReviewAnalyzer(checkpoint_dir=tmp_path)
    assert analyzer.is_ready() is False

    (tmp_path / "metadata.json").write_text(json.dumps({"encoder_model_name": "demo"}))
    (tmp_path / "model.pt").write_text("stub")
    (tmp_path / "encoder").mkdir()
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert analyzer.is_ready() is True


def test_analyzer_returns_explanation_artifacts() -> None:
    class DummyTokenizer:
        def __call__(self, text: str, **kwargs):
            raw_ids = [101, 11, 12, 13, 14, 15, 102]
            if kwargs.get("return_tensors") == "pt":
                token_ids = raw_ids[: kwargs["max_length"]]
                return {
                    "input_ids": torch.tensor([token_ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }
            return {"input_ids": raw_ids}

    class DummyModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                sentiment_logits=torch.tensor([[1.2, -0.5, 2.4]]),
                authenticity_logits=torch.tensor([[0.1, 1.7]]),
            )

    analyzer = ReviewAnalyzer(checkpoint_dir=Path("models/latest"))
    analyzer.device = torch.device("cpu")
    analyzer.tokenizer = DummyTokenizer()
    analyzer.model = DummyModel()
    analyzer.metadata = {
        "encoder_model_name": "demo-export",
        "sentiment_labels": ["negative", "neutral", "positive"],
        "authenticity_labels": ["authentic", "fake"],
        "max_length": 4,
    }

    result = analyzer.analyze("Helpful review text for inference.")

    assert result["sentiment_label"] == "positive"
    assert result["authenticity_label"] == "fake"
    assert result["model_name"] == "demo-export"
    assert result["explanation"]["token_count"] == 4
    assert result["explanation"]["max_length"] == 4
    assert result["explanation"]["truncated"] is True
    assert [item["label"] for item in result["explanation"]["sentiment_top_probabilities"]] == [
        "positive",
        "negative",
        "neutral",
    ]
    assert [item["label"] for item in result["explanation"]["authenticity_top_probabilities"]] == [
        "fake",
        "authentic",
    ]
    assert result["explanation"]["sentiment_top_probabilities"][0]["probability"] == pytest.approx(
        0.737345,
        rel=1e-5,
    )
    assert any(
        "Only the first 4 tokens were scored" in note
        for note in result["explanation"]["notes"]
    )
