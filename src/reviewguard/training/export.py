from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_export_dir(path: str | Path) -> Path:
    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return target
