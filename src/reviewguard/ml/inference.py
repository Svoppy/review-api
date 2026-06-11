from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    def _margin(probabilities: list[float]) -> float:
        ranked = sorted(probabilities, reverse=True)
        if len(ranked) < 2:
            return float(ranked[0]) if ranked else 0.0
        return float(ranked[0] - ranked[1])

    @classmethod
    def _build_risk_flags(
        cls,
        *,
        truncated: bool,
        sentiment_probs: list[float],
        authenticity_probs: list[float],
    ) -> list[str]:
        flags: list[str] = []
        if truncated:
            flags.append("truncated_input")
        if cls._margin(sentiment_probs) < 0.15:
            flags.append("low_margin_sentiment")
        if cls._margin(authenticity_probs) < 0.15:
            flags.append("low_margin_authenticity")
        return flags

    @staticmethod
    def _build_notes(
        *,
        max_length: int,
        raw_token_count: int,
        truncated: bool,
        risk_flags: list[str],
    ) -> list[str]:
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

        if "low_margin_sentiment" in risk_flags or "low_margin_authenticity" in risk_flags:
            notes.append(
                "At least one task has a narrow top-1 vs top-2 probability margin, so this prediction should be treated as borderline."
            )

        return notes

    def checkpoint_report(self) -> dict[str, Any] | None:
        report_path = self.checkpoint_dir / "train_report.json"
        if not report_path.exists():
            return None
        return json.loads(report_path.read_text())

    def research_context(self) -> dict[str, Any]:
        report = self.checkpoint_report() or {}
        input_summary = report.get("input_summary", {})
        split_audit = report.get("split_audit", {})
        robustness = report.get("robustness", {})
        warnings = list(split_audit.get("warnings", []))
        for task in ("sentiment", "authenticity"):
            warnings.extend(split_audit.get("tasks", {}).get(task, {}).get("warnings", []))

        robustness_summary: dict[str, Any] = {}
        for task, slice_payload in robustness.items():
            source_report = slice_payload.get("source", {})
            robustness_summary[task] = {
                "mean_source_macro_f1": source_report.get("macro_f1_mean_across_slices"),
                "worst_source_macro_f1": source_report.get("worst_slice_macro_f1"),
                "source_gap": source_report.get("robustness_gap"),
            }

        return {
            "scope": "pilot" if report else "checkpoint_only",
            "training_records": input_summary.get("records"),
            "sentiment_labeled": input_summary.get("sentiment_labeled"),
            "authenticity_labeled": input_summary.get("authenticity_labeled"),
            "sources": input_summary.get("sources", {}),
            "warnings": warnings[:6],
            "robustness_summary": robustness_summary,
        }

    def is_ready(self) -> bool:
        return (
            self.checkpoint_dir.exists()
            and (self.checkpoint_dir / "metadata.json").exists()
            and (self.checkpoint_dir / "model.pt").exists()
            and (self.checkpoint_dir / "encoder").is_dir()
            and (self.checkpoint_dir / "tokenizer_config.json").exists()
        )

    def checkpoint_metadata(self) -> dict[str, object] | None:
        metadata_path = self.checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text())

    def effective_model_name(self) -> str:
        metadata = self.checkpoint_metadata()
        if metadata and metadata.get("encoder_model_name"):
            return str(metadata["encoder_model_name"])
        return self.model_name

    def load(self) -> None:
        if not self.is_ready():
            raise ModelNotReadyError(
                "No trained checkpoint found yet. Train and export a model into models/latest first."
            )

        metadata = self.checkpoint_metadata() or {}
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
        risk_flags = self._build_risk_flags(
            truncated=truncated,
            sentiment_probs=sentiment_probs,
            authenticity_probs=authenticity_probs,
        )

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
                    risk_flags=risk_flags,
                ),
                "risk_flags": risk_flags,
                "token_count": token_count,
                "max_length": max_length,
                "truncated": truncated,
                "sentiment_margin": self._margin(sentiment_probs),
                "authenticity_margin": self._margin(authenticity_probs),
            },
            "research_context": self.research_context(),
        }
