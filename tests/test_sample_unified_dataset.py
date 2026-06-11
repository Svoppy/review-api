from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sampling_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "sample_unified_dataset.py"
    spec = importlib.util.spec_from_file_location("sample_unified_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_source_caps_normalizes_common_aliases() -> None:
    module = _load_sampling_module()

    caps = module._parse_source_caps(
        ["perekrestok_reviews=10", "ru_reviews=5", "maide_up=3"]
    )

    assert caps == {
        "perekrestok": 10,
        "rureviews": 5,
        "maide_up": 3,
    }
