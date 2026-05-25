import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, "src")

from reviewguard.ml.inference import ReviewAnalyzer


def test_analyzer_readiness_requires_export_files(tmp_path: Path) -> None:
    analyzer = ReviewAnalyzer(checkpoint_dir=tmp_path)
    assert analyzer.is_ready() is False

    (tmp_path / "metadata.json").write_text(json.dumps({"encoder_model_name": "demo"}))
    (tmp_path / "model.pt").write_text("stub")
    assert analyzer.is_ready() is True
