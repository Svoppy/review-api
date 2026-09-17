from __future__ import annotations

import pytest

from reviewguard.training.runtime import TrainingDevice, require_accelerator, resolve_training_device


def test_cpu_is_a_valid_explicit_device() -> None:
    device = resolve_training_device("cpu")

    assert device.requested == "cpu"
    assert device.resolved == "cpu"
    assert device.accelerator_available is False


def test_require_accelerator_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA or Apple Metal"):
        require_accelerator(
            TrainingDevice(requested="auto", resolved="cpu", accelerator_available=False),
            experiment_name="article20k",
        )
