from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from reviewguard.config import settings
from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.ml.modeling import MultiTaskTransformer


class ModelNotReadyError(RuntimeError):
    pass


class ReviewAnalyzer:
    def __init__(self, checkpoint_dir: Path | None = None, model_name: str | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
        self.model_name = model_name or settings.model_name
        self.device = torch.device(settings.device)

    @staticmethod
    def _top_probabilities(labels: list[str], probabilities: list[float], top_k: int = 3) -> list[dict[str, object]]:
        ranked_indices = sorted(
            range(len(probabilities)),
            key=probabilities.__getitem__,
            reverse=True,
        )[: min(top_k, len(probabilities))]
        return [
            {
                "label": labels[index],
                "probability": probabilities[index],
            }
            for index in ranked_indices
        ]

    @staticmethod
    def _build_notes(*, max_length: int, raw_token_count: int, truncated: bool) -> list[str]:
        notes = [
            "Probabilities are ranked independently for sentiment and authenticity, and each task's list sums to 1.0.",
            "These scores show the model's relative preference inside each task and should not be treated as calibrated certainty.",
        ]

        if truncated:
            notes.append(
                f"Only the first {max_length} tokens were scored because the review exceeded the model limit ({raw_token_count} tokens before truncation)."
            )
        else:
            notes.append(
                f"The full review fit within the model limit ({raw_token_count}/{max_length} tokens used)."
            )

        return notes

    def is_ready(self) -> bool:
        return (
            self.checkpoint_dir.exists()
            and (self.checkpoint_dir / "metadata.json").exists()
            and (self.checkpoint_dir / "model.pt").exists()
            and (self.checkpoint_dir / "encoder").is_dir()
            and (self.checkpoint_dir / "tokenizer_config.json").exists()
        )

    def load(self) -> None:
        if not self.is_ready():
            raise ModelNotReadyError(
                "No trained checkpoint found yet. Train and export a model into models/latest first."
            )

        metadata = json.loads((self.checkpoint_dir / "metadata.json").read_text())
        self.metadata = metadata
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.model, _ = MultiTaskTransformer.from_exported_checkpoint(self.checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text: str) -> dict[str, object]:
        if not hasattr(self, "model"):
            self.load()

        max_length = int(self.metadata.get("max_length", settings.max_length))
        raw_token_count = len(
            self.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )
        token_count = int(encoded["input_ids"].shape[-1])
        truncated = raw_token_count > max_length
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**encoded)
            sentiment_probs = torch.softmax(outputs.sentiment_logits, dim=-1)[0].cpu().tolist()
            authenticity_probs = torch.softmax(outputs.authenticity_logits, dim=-1)[0].cpu().tolist()

        sentiment_labels = self.metadata.get("sentiment_labels", SENTIMENT_LABELS)
        authenticity_labels = self.metadata.get("authenticity_labels", AUTHENTICITY_LABELS)

        sentiment_index = max(range(len(sentiment_probs)), key=sentiment_probs.__getitem__)
        authenticity_index = max(range(len(authenticity_probs)), key=authenticity_probs.__getitem__)

        return {
            "sentiment_label": sentiment_labels[sentiment_index],
            "sentiment_confidence": sentiment_probs[sentiment_index],
            "authenticity_label": authenticity_labels[authenticity_index],
            "authenticity_confidence": authenticity_probs[authenticity_index],
            "model_name": self.metadata.get("encoder_model_name", self.model_name),
            "explanation": {
                "sentiment_top_probabilities": self._top_probabilities(
                    sentiment_labels,
                    sentiment_probs,
                ),
                "authenticity_top_probabilities": self._top_probabilities(
                    authenticity_labels,
                    authenticity_probs,
                ),
                "notes": self._build_notes(
                    max_length=max_length,
                    raw_token_count=raw_token_count,
                    truncated=truncated,
                ),
                "token_count": token_count,
                "max_length": max_length,
                "truncated": truncated,
            },
        }
