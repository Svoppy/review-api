from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from reviewguard.config import settings
from reviewguard.ml.modeling import MultiTaskTransformer


class ModelNotReadyError(RuntimeError):
    pass


class ReviewAnalyzer:
    def __init__(self, checkpoint_dir: Path | None = None, model_name: str | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
        self.model_name = model_name or settings.model_name

    def is_ready(self) -> bool:
        return self.checkpoint_dir.exists()

    def load(self) -> None:
        if not self.is_ready():
            raise ModelNotReadyError(
                "No trained checkpoint found yet. Train and export a model into models/latest first."
            )

        # Placeholder loading path for the next milestone.
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.model = MultiTaskTransformer(model_name=str(self.checkpoint_dir))
        self.model.eval()

    def analyze(self, text: str) -> dict[str, object]:
        if not hasattr(self, "model"):
            self.load()

        raise ModelNotReadyError(
            "Inference pipeline scaffolded, but prediction logic will be enabled after checkpoint export."
        )
