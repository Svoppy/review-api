from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_preflight_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "preflight_article_experiment.py"
    spec = importlib.util.spec_from_file_location("preflight_article_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_reports_ready_for_balanced_multitask_fixture(tmp_path: Path) -> None:
    records = []
    for index in range(90):
        records.append(
            {
                "record_id": str(index),
                "text": f"review {index}",
                "source": "maide_up" if index % 2 else "rureviews",
                "language": "en" if index % 2 else "ru",
                "domain": "hospitality" if index % 2 else "ecommerce",
                "sentiment_label": ["negative", "neutral", "positive"][index % 3],
                "authenticity_label": "fake" if index % 4 == 1 else ("authentic" if index % 4 == 3 else None),
            }
        )

    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_name: demo-model\npooling: cls\nhead_type: mlp\n",
        encoding="utf-8",
    )

    module = _load_preflight_module()
    report = module.build_preflight_report(
        input_path=input_path,
        config_path=config_path,
        split_seed=7,
        min_sentiment_class_support=1,
        min_authenticity_class_support=1,
    )

    assert report["ready_to_train"] is True
    assert report["checks"]["zero_cross_split_overlap"] is True
    assert report["execution_checks"]["accelerator_requirement_satisfied"] is True
    assert report["config"]["head_type"] == "mlp"
