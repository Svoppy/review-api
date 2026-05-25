from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.training.export import ensure_export_dir, write_json
from reviewguard.training.metrics import compute_multitask_metrics


@dataclass(frozen=True)
class BaselineConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_features: int = 50_000
    max_iter: int = 1_000
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42


class ClassicalBaselineTrainer:
    """Train one TF-IDF + LogisticRegression model per available task."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.models: dict[str, Pipeline] = {}

    def fit(self, records: list[dict[str, Any]]) -> "ClassicalBaselineTrainer":
        self.models = {}
        for task, label_field in (
            ("sentiment", "sentiment_label"),
            ("authenticity", "authenticity_label"),
        ):
            task_records = [record for record in records if record.get(label_field) is not None]
            labels = [record[label_field] for record in task_records]
            if len(task_records) < 2 or len(set(labels)) < 2:
                continue

            pipeline = Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=self.config.ngram_range,
                            min_df=self.config.min_df,
                            max_features=self.config.max_features,
                            strip_accents="unicode",
                            lowercase=True,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=self.config.max_iter,
                            class_weight=self.config.class_weight,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            )
            pipeline.fit([record["text"] for record in task_records], labels)
            self.models[task] = pipeline

        return self

    def predict(self, records: list[dict[str, Any]]) -> dict[str, list[str] | None]:
        outputs: dict[str, list[str] | None] = {
            "sentiment": None,
            "authenticity": None,
        }
        texts = [record["text"] for record in records]
        for task, model in self.models.items():
            outputs[task] = model.predict(texts).tolist()
        return outputs

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        predictions = self.predict(records)
        return compute_multitask_metrics(
            records,
            sentiment_predictions=predictions["sentiment"],
            authenticity_predictions=predictions["authenticity"],
            sentiment_labels=SENTIMENT_LABELS,
            authenticity_labels=AUTHENTICITY_LABELS,
        )

    def export(self, export_dir: str | Path) -> Path:
        target_dir = ensure_export_dir(export_dir)
        for task, model in self.models.items():
            with (target_dir / f"{task}.pkl").open("wb") as file_obj:
                pickle.dump(model, file_obj)

        manifest = {
            "artifact_type": "classical_baseline",
            "model_family": "tfidf_logistic_regression",
            "tasks": sorted(self.models.keys()),
            "files": {task: f"{task}.pkl" for task in self.models},
            "label_spaces": {
                "sentiment": SENTIMENT_LABELS,
                "authenticity": AUTHENTICITY_LABELS,
            },
            "config": {
                "ngram_range": list(self.config.ngram_range),
                "min_df": self.config.min_df,
                "max_features": self.config.max_features,
                "max_iter": self.config.max_iter,
                "class_weight": self.config.class_weight,
                "random_state": self.config.random_state,
            },
        }
        write_json(target_dir / "manifest.json", manifest)
        return target_dir
