from __future__ import annotations

from dataclasses import dataclass

from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS


DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_MAX_LENGTH = 256


TASK_LABELS = {
    "sentiment": SENTIMENT_LABELS,
    "authenticity": AUTHENTICITY_LABELS,
}


@dataclass(frozen=True)
class SingleTaskTrainingConfig:
    task: str
    labels: list[str]
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 1
    dropout: float = 0.1
    device: str = "cpu"
    random_state: int = 42
