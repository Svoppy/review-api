from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass
class MultiTaskOutput:
    sentiment_logits: torch.Tensor
    authenticity_logits: torch.Tensor
    loss: torch.Tensor | None = None


class MultiTaskTransformer(nn.Module):
    """Shared-encoder multitask classifier for sentiment and authenticity."""

    def __init__(
        self,
        model_name: str,
        sentiment_num_labels: int = 3,
        authenticity_num_labels: int = 2,
        dropout: float = 0.1,
        sentiment_loss_weight: float = 1.0,
        authenticity_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, sentiment_num_labels)
        self.authenticity_head = nn.Linear(hidden_size, authenticity_num_labels)
        self.sentiment_loss_weight = sentiment_loss_weight
        self.authenticity_loss_weight = authenticity_loss_weight
        self.encoder_model_name = model_name

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mean pooling is more stable across encoder families than relying on CLS conventions.
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        sentiment_labels: torch.Tensor | None = None,
        authenticity_labels: torch.Tensor | None = None,
    ) -> MultiTaskOutput:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        try:
            encoded = self.encoder(**encoder_kwargs)
        except TypeError:
            encoder_kwargs.pop("token_type_ids", None)
            encoded = self.encoder(**encoder_kwargs)
        pooled = self.dropout(self._pool(encoded.last_hidden_state, attention_mask))

        sentiment_logits = self.sentiment_head(pooled)
        authenticity_logits = self.authenticity_head(pooled)

        loss = None
        if sentiment_labels is not None or authenticity_labels is not None:
            parts = []
            if sentiment_labels is not None:
                parts.append(
                    self.sentiment_loss_weight
                    * nn.functional.cross_entropy(sentiment_logits, sentiment_labels)
                )
            if authenticity_labels is not None:
                parts.append(
                    self.authenticity_loss_weight
                    * nn.functional.cross_entropy(authenticity_logits, authenticity_labels)
                )
            loss = torch.stack(parts).sum() if parts else None

        return MultiTaskOutput(
            sentiment_logits=sentiment_logits,
            authenticity_logits=authenticity_logits,
            loss=loss,
        )

    def export_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        tokenizer,
        sentiment_labels: list[str],
        authenticity_labels: list[str],
        max_length: int,
    ) -> None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        encoder_dir = checkpoint_path / "encoder"
        self.encoder.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path / "model.pt")

        metadata = {
            "encoder_model_name": self.encoder_model_name,
            "encoder_dir": "encoder",
            "sentiment_labels": sentiment_labels,
            "authenticity_labels": authenticity_labels,
            "max_length": max_length,
        }
        (checkpoint_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def from_exported_checkpoint(cls, checkpoint_dir: str | Path) -> tuple[MultiTaskTransformer, dict]:
        checkpoint_path = Path(checkpoint_dir)
        metadata = json.loads((checkpoint_path / "metadata.json").read_text())
        encoder_dir = checkpoint_path / metadata.get("encoder_dir", "encoder")

        model = cls(
            model_name=str(encoder_dir),
            sentiment_num_labels=len(metadata["sentiment_labels"]),
            authenticity_num_labels=len(metadata["authenticity_labels"]),
        )
        state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model, metadata
