from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from reviewguard.config import settings
from reviewguard.ml.datasets import AUTHENTICITY_LABELS, SENTIMENT_LABELS
from reviewguard.ml.modeling import MultiTaskTransformer
from reviewguard.training.export import ensure_export_dir, write_json
from reviewguard.training.metrics import compute_multitask_metrics


def seed_training_runtime(random_state: int) -> None:
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


@dataclass(frozen=True)
class MultitaskTrainingConfig:
    model_name: str = settings.model_name
    max_length: int = settings.max_length
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 4
    dropout: float = 0.1
    sentiment_loss_weight: float = 1.0
    authenticity_loss_weight: float = 1.0
    class_weight_mode: str = "balanced"
    train_sampler: str = "source_balanced"
    early_stopping_patience: int | None = 2
    device: str = "cpu"
    random_state: int = 42


class UnifiedReviewDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer: AutoTokenizer,
        max_length: int,
        sentiment_labels: list[str] | None = None,
        authenticity_labels: list[str] | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiment_to_id = {
            label: index for index, label in enumerate(sentiment_labels or SENTIMENT_LABELS)
        }
        self.authenticity_to_id = {
            label: index for index, label in enumerate(authenticity_labels or AUTHENTICITY_LABELS)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        sentiment_label = record.get("sentiment_label")
        authenticity_label = record.get("authenticity_label")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sentiment_labels": torch.tensor(
                self.sentiment_to_id[sentiment_label] if sentiment_label is not None else -100,
                dtype=torch.long,
            ),
            "authenticity_labels": torch.tensor(
                self.authenticity_to_id[authenticity_label]
                if authenticity_label is not None
                else -100,
                dtype=torch.long,
            ),
        }


class MultitaskTrainingScaffold:
    """Lightweight multitask trainer with export metadata for downstream app wiring."""

    def __init__(self, config: MultitaskTrainingConfig | None = None) -> None:
        self.config = config or MultitaskTrainingConfig()
        seed_training_runtime(self.config.random_state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = MultiTaskTransformer(
            model_name=self.config.model_name,
            sentiment_num_labels=len(SENTIMENT_LABELS),
            authenticity_num_labels=len(AUTHENTICITY_LABELS),
            dropout=self.config.dropout,
            sentiment_loss_weight=self.config.sentiment_loss_weight,
            authenticity_loss_weight=self.config.authenticity_loss_weight,
        )
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def _make_loader(self, records: list[dict[str, Any]], shuffle: bool) -> DataLoader:
        dataset = UnifiedReviewDataset(
            records,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        generator = None
        sampler = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.config.random_state)
            if self.config.train_sampler == "source_balanced":
                source_counts = Counter(str(record.get("source") or "unknown") for record in records)
                sample_weights = [
                    1.0 / source_counts[str(record.get("source") or "unknown")] for record in records
                ]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=generator,
                )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            generator=generator,
        )

    def _task_class_weights(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor | None]:
        if self.config.class_weight_mode != "balanced":
            return {"sentiment": None, "authenticity": None}

        outputs: dict[str, torch.Tensor | None] = {}
        for task, field, labels in (
            ("sentiment", "sentiment_label", SENTIMENT_LABELS),
            ("authenticity", "authenticity_label", AUTHENTICITY_LABELS),
        ):
            counts = Counter(record[field] for record in records if record.get(field) is not None)
            if len(counts) < 2:
                outputs[task] = None
                continue
            total = sum(counts.values())
            weights = [
                total / (len(labels) * counts.get(label, total)) if counts.get(label, 0) > 0 else 0.0
                for label in labels
            ]
            outputs[task] = torch.tensor(weights, dtype=torch.float)
        return outputs

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        class_weights: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, int]]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        parts: list[torch.Tensor] = []
        counts = {"sentiment": 0, "authenticity": 0}

        sentiment_mask = batch["sentiment_labels"] != -100
        if sentiment_mask.any():
            parts.append(
                self.config.sentiment_loss_weight
                * torch.nn.functional.cross_entropy(
                    outputs.sentiment_logits[sentiment_mask],
                    batch["sentiment_labels"][sentiment_mask],
                    weight=(
                        class_weights["sentiment"].to(self.device)
                        if class_weights["sentiment"] is not None
                        else None
                    ),
                )
            )
            counts["sentiment"] = int(sentiment_mask.sum().item())

        authenticity_mask = batch["authenticity_labels"] != -100
        if authenticity_mask.any():
            parts.append(
                self.config.authenticity_loss_weight
                * torch.nn.functional.cross_entropy(
                    outputs.authenticity_logits[authenticity_mask],
                    batch["authenticity_labels"][authenticity_mask],
                    weight=(
                        class_weights["authenticity"].to(self.device)
                        if class_weights["authenticity"] is not None
                        else None
                    ),
                )
            )
            counts["authenticity"] = int(authenticity_mask.sum().item())

        if not parts:
            raise ValueError("Encountered a batch with no task labels.")

        return torch.stack(parts).sum(), counts

    def fit(
        self,
        train_records: list[dict[str, Any]],
        *,
        valid_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        class_weights = self._task_class_weights(train_records)
        train_loader = self._make_loader(train_records, shuffle=True)
        history: list[dict[str, Any]] = []
        best_validation_score: float | None = None
        best_state_dict: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            labeled_counts = {"sentiment": 0, "authenticity": 0}

            for batch in train_loader:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                optimizer.zero_grad()
                loss, counts = self._compute_batch_loss(batch, class_weights=class_weights)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1
                labeled_counts["sentiment"] += counts["sentiment"]
                labeled_counts["authenticity"] += counts["authenticity"]

            epoch_summary: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batch_count, 1),
                "labeled_examples": labeled_counts,
            }
            if valid_records:
                validation_metrics = self.evaluate(valid_records)
                epoch_summary["validation"] = validation_metrics
                macro_scores = [
                    float(task_metrics["macro_f1"])
                    for task_metrics in validation_metrics.values()
                    if task_metrics
                ]
                validation_score = float(sum(macro_scores) / len(macro_scores)) if macro_scores else 0.0
                epoch_summary["validation_score"] = validation_score
                if best_validation_score is None or validation_score > best_validation_score:
                    best_validation_score = validation_score
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    epoch_summary["is_best_epoch"] = True
                else:
                    epochs_without_improvement += 1
            history.append(epoch_summary)

            if (
                valid_records
                and self.config.early_stopping_patience is not None
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.history = history
        return {
            "history": history,
            "model_name": self.config.model_name,
            "epochs": len(history),
            "best_validation_mean_macro_f1": best_validation_score,
        }

    def predict(self, records: list[dict[str, Any]]) -> dict[str, list[str]]:
        loader = self._make_loader(records, shuffle=False)
        self.model.eval()
        sentiment_predictions: list[str] = []
        authenticity_predictions: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                sentiment_ids = outputs.sentiment_logits.argmax(dim=-1).tolist()
                authenticity_ids = outputs.authenticity_logits.argmax(dim=-1).tolist()
                sentiment_predictions.extend(SENTIMENT_LABELS[index] for index in sentiment_ids)
                authenticity_predictions.extend(
                    AUTHENTICITY_LABELS[index] for index in authenticity_ids
                )

        return {
            "sentiment": sentiment_predictions,
            "authenticity": authenticity_predictions,
        }

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
        self.model.export_checkpoint(
            target_dir,
            tokenizer=self.tokenizer,
            sentiment_labels=list(SENTIMENT_LABELS),
            authenticity_labels=list(AUTHENTICITY_LABELS),
            max_length=self.config.max_length,
        )

        manifest = {
            "artifact_type": "multitask_transformer",
            "model_name": self.config.model_name,
            "checkpoint_files": {
                "encoder_dir": "encoder/",
                "tokenizer": "tokenizer files in export root",
                "state_dict": "model.pt",
                "metadata": "metadata.json",
            },
            "label_spaces": {
                "sentiment": SENTIMENT_LABELS,
                "authenticity": AUTHENTICITY_LABELS,
            },
            "training_config": {
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "epochs": self.config.epochs,
                "dropout": self.config.dropout,
                "sentiment_loss_weight": self.config.sentiment_loss_weight,
                "authenticity_loss_weight": self.config.authenticity_loss_weight,
                "class_weight_mode": self.config.class_weight_mode,
                "train_sampler": self.config.train_sampler,
                "early_stopping_patience": self.config.early_stopping_patience,
                "device": self.config.device,
                "random_state": self.config.random_state,
            },
            "history": getattr(self, "history", []),
            "consumer_note": "The FastAPI app loads this exported directory directly via ReviewAnalyzer.",
        }
        write_json(target_dir / "manifest.json", manifest)
        return target_dir
