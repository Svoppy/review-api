from __future__ import annotations

from dataclasses import dataclass

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
        sentiment_labels: torch.Tensor | None = None,
        authenticity_labels: torch.Tensor | None = None,
    ) -> MultiTaskOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
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
