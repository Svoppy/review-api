from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reviewguard.data import load_unified_records
from reviewguard.ml.modeling import MultiTaskTransformer
from reviewguard.training.multitask import UnifiedReviewDataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe gradient conflict between sentiment and authenticity on the shared encoder."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    return parser.parse_args()


def flatten_grads(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parts.append(parameter.grad.detach().flatten().cpu())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = float(left.norm() * right.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def main() -> int:
    args = parse_args()
    model, metadata = MultiTaskTransformer.from_exported_checkpoint(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model.train()
    records = load_unified_records(args.input)
    records = [
        record
        for record in records
        if record.get("sentiment_label") is not None and record.get("authenticity_label") is not None
    ]
    dataset = UnifiedReviewDataset(
        records,
        tokenizer=tokenizer,
        max_length=int(metadata["max_length"]),
        sentiment_labels=list(metadata["sentiment_labels"]),
        authenticity_labels=list(metadata["authenticity_labels"]),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    encoder_parameters = list(model.encoder.parameters())
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break

        sentiment_mask = batch["sentiment_labels"] != -100
        authenticity_mask = batch["authenticity_labels"] != -100
        if not sentiment_mask.any() or not authenticity_mask.any():
            continue

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        sentiment_loss = torch.nn.functional.cross_entropy(
            outputs.sentiment_logits[sentiment_mask],
            batch["sentiment_labels"][sentiment_mask],
        )
        sentiment_loss.backward(retain_graph=True)
        sentiment_grad = flatten_grads(encoder_parameters)

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        authenticity_loss = torch.nn.functional.cross_entropy(
            outputs.authenticity_logits[authenticity_mask],
            batch["authenticity_labels"][authenticity_mask],
        )
        authenticity_loss.backward()
        authenticity_grad = flatten_grads(encoder_parameters)

        rows.append(
            {
                "batch_index": batch_index,
                "sentiment_grad_norm": float(sentiment_grad.norm()),
                "authenticity_grad_norm": float(authenticity_grad.norm()),
                "gradient_cosine_similarity": cosine_similarity(sentiment_grad, authenticity_grad),
                "sentiment_examples": int(sentiment_mask.sum()),
                "authenticity_examples": int(authenticity_mask.sum()),
            }
        )

    mean_cosine = sum(row["gradient_cosine_similarity"] for row in rows) / len(rows) if rows else 0.0
    payload = {
        "input": str(args.input),
        "checkpoint_dir": str(args.checkpoint_dir),
        "batches_evaluated": len(rows),
        "mean_gradient_cosine_similarity": mean_cosine,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote gradient probe to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
