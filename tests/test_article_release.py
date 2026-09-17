from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_release_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "run_article_release.py"
    spec = importlib.util.spec_from_file_location("run_article_release", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_commands_cover_preflight_training_statistics_and_ablation() -> None:
    release = _load_release_module()
    commands = release.build_stage_commands(
        python_executable="python",
        input_path=Path("data.jsonl"),
        config_snapshot=Path("reports/run/config.used.yaml"),
        preflight_report=Path("reports/preflight.json"),
        output_root=Path("models/run"),
        report_root=Path("reports/run"),
        statistics_root=Path("reports/statistics"),
        ablation_root=Path("reports/ablation"),
        split_seed=42,
        train_seeds=[11, 21],
        bootstrap_samples=500,
        randomization_samples=500,
        skip_existing=True,
    )

    assert [name for name, _ in commands] == [
        "preflight",
        "multiseed_training",
        "statistical_analysis",
        "task_ablation",
    ]
    assert "--strict" in commands[0][1]
    assert commands[1][1][-1] == "--skip-existing"
    assert commands[2][1][commands[2][1].index("--train-seeds") + 1 :][:2] == ["11", "21"]


def test_release_seed_resolution_rejects_duplicates() -> None:
    release = _load_release_module()

    with pytest.raises(ValueError, match="unique"):
        release.resolve_train_seeds({"experiment": {"train_seeds": [11, 11]}}, None)
