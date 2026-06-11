from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from reviewguard.training.export import ensure_export_dir, write_json
from reviewguard.training.metrics import compute_task_metrics, label_field_for_task
from reviewguard.training.single_task_config import (
    TASK_LABELS,
    SingleTaskTrainingConfig,
)


def seed_training_runtime(random_state: int) -> None:
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def labeled_records_for_task(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    label_field = label_field_for_task(task)
    return [record for record in records if record.get(label_field) is not None]


class SingleTaskReviewDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer: AutoTokenizer,
        max_length: int,
        label_field: str,
        label_to_id: dict[str, int],
        include_labels: bool,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_field = label_field
        self.label_to_id = label_to_id
        self.include_labels = include_labels

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
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.include_labels:
            item["labels"] = torch.tensor(
                self.label_to_id[record[self.label_field]],
                dtype=torch.long,
            )
        return item


class SingleTaskTransformerTrainer:
    """Train a single-head Transformer baseline for one classification task."""

    def __init__(self, config: SingleTaskTrainingConfig) -> None:
        if config.task not in TASK_LABELS:
            raise ValueError(f"Unsupported task: {config.task}")

        self.config = config
        self.task = config.task
        self.labels = list(config.labels)
        self.label_field = label_field_for_task(self.task)
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}

        # Seed before model construction so classifier-head initialization is reproducible.
        seed_training_runtime(self.config.random_state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=len(self.labels),
        )
        if hasattr(model_config, "classifier_dropout"):
            model_config.classifier_dropout = self.config.dropout
        if hasattr(model_config, "hidden_dropout_prob"):
            model_config.hidden_dropout_prob = self.config.dropout
        if hasattr(model_config, "attention_probs_dropout_prob"):
            model_config.attention_probs_dropout_prob = self.config.dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=model_config,
        )
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def _make_loader(
        self,
        records: list[dict[str, Any]],
        *,
        shuffle: bool,
        include_labels: bool,
    ) -> DataLoader:
        dataset = SingleTaskReviewDataset(
            records,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            label_field=self.label_field,
            label_to_id=self.label_to_id,
            include_labels=include_labels,
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _class_weight_tensor(self, records: list[dict[str, Any]]) -> torch.Tensor | None:
        if self.config.class_weight_mode != "balanced":
            return None
        counts = Counter(record[self.label_field] for record in records)
        if len(counts) < 2:
            return None
        total = sum(counts.values())
        weights = [
            total / (len(self.labels) * counts.get(label, total)) if counts.get(label, 0) > 0 else 0.0
            for label in self.labels
        ]
        return torch.tensor(weights, dtype=torch.float)

    def fit(
        self,
        train_records: list[dict[str, Any]],
        *,
        valid_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        labeled_train_records = labeled_records_for_task(train_records, self.task)
        if not labeled_train_records:
            raise ValueError(
                f"No labeled examples available for task '{self.task}' in the training split."
            )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        class_weights = self._class_weight_tensor(labeled_train_records)
        train_loader = self._make_loader(
            labeled_train_records,
            shuffle=True,
            include_labels=True,
        )
        history: list[dict[str, Any]] = []
        best_validation_score: float | None = None
        best_state_dict: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits,
                    batch["labels"],
                    weight=class_weights.to(self.device) if class_weights is not None else None,
                )
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            epoch_summary: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batch_count, 1),
                "labeled_examples": len(labeled_train_records),
            }
            if valid_records:
                validation_metrics = self.evaluate(valid_records)
                epoch_summary["validation"] = validation_metrics
                task_metrics = validation_metrics.get(self.task, {})
                validation_score = float(task_metrics.get("macro_f1", 0.0)) if task_metrics else 0.0
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
            "task": self.task,
            "epochs": len(history),
            "train_labeled_examples": len(labeled_train_records),
            "best_validation_macro_f1": best_validation_score,
        }

    def predict(self, records: list[dict[str, Any]]) -> list[str]:
        if not records:
            return []

        loader = self._make_loader(records, shuffle=False, include_labels=False)
        self.model.eval()
        predictions: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                predicted_ids = outputs.logits.argmax(dim=-1).tolist()
                predictions.extend(self.id_to_label[index] for index in predicted_ids)

        return predictions

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        labeled_records = labeled_records_for_task(records, self.task)
        if not labeled_records:
            return {}

        return compute_task_metrics(
            labeled_records,
            predictions=self.predict(labeled_records),
            task=self.task,
            labels=self.labels,
        )

    def export(self, export_dir: str | Path) -> Path:
        target_dir = ensure_export_dir(export_dir)
        self.model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)

        metadata = {
            "task": self.task,
            "label_field": self.label_field,
            "labels": self.labels,
            "encoder_model_name": self.config.model_name,
            "max_length": self.config.max_length,
        }
        write_json(target_dir / "metadata.json", metadata)

        manifest = {
            "artifact_type": "single_task_transformer",
            "task": self.task,
            "model_name": self.config.model_name,
            "label_space": self.labels,
            "checkpoint_files": {
                "weights": "model.safetensors or pytorch_model.bin",
                "config": "config.json",
                "tokenizer": "tokenizer files in export root",
                "metadata": "metadata.json",
            },
            "training_config": {
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "epochs": self.config.epochs,
                "dropout": self.config.dropout,
                "class_weight_mode": self.config.class_weight_mode,
                "early_stopping_patience": self.config.early_stopping_patience,
                "device": self.config.device,
                "random_state": self.config.random_state,
            },
            "history": getattr(self, "history", []),
            "consumer_note": "This export contains one task-specific Transformer classifier.",
        }
        write_json(target_dir / "manifest.json", manifest)
        return target_dir
