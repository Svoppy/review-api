"""Runtime selection utilities for reproducible Transformer experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingDevice:
    requested: str
    resolved: str
    accelerator_available: bool


def resolve_training_device(requested: str = "auto") -> TrainingDevice:
    """Resolve a portable training device without silently falling back from an explicit request."""

    normalized = requested.strip().lower()
    cuda_available = torch.cuda.is_available()
    mps_available = bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())

    if normalized == "auto":
        resolved = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif normalized == "cuda":
        if not cuda_available:
            raise RuntimeError("device='cuda' was requested, but CUDA is not available.")
        resolved = "cuda"
    elif normalized == "mps":
        if not mps_available:
            raise RuntimeError("device='mps' was requested, but Apple Metal is not available.")
        resolved = "mps"
    elif normalized == "cpu":
        resolved = "cpu"
    else:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")

    return TrainingDevice(
        requested=normalized,
        resolved=resolved,
        accelerator_available=resolved in {"cuda", "mps"},
    )


def require_accelerator(device: TrainingDevice, *, experiment_name: str) -> None:
    if not device.accelerator_available:
        raise RuntimeError(
            f"{experiment_name} requires CUDA or Apple Metal for article-grade Transformer training; "
            "CPU execution is reserved for tests and smoke runs."
        )
