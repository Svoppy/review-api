from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from reviewguard.api.main import app


def load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of demo examples.")
    return payload


def probability_summary(items: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{item['label']}={float(item['probability']):.3f}" for item in items
    )


def render_markdown(
    examples: list[dict[str, Any]],
    health: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Defense Demo Results")
    lines.append("")
    lines.append("## Active checkpoint")
    lines.append("")
    lines.append(f"- status: `{health['status']}`")
    lines.append(f"- model_ready: `{health['model_ready']}`")
    lines.append(f"- model_name: `{health['model_name']}`")
    lines.append(f"- checkpoint_dir: `{health['checkpoint_dir']}`")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| ID | Language | Sentiment | Sentiment confidence | Authenticity | Authenticity confidence |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")

    for example, result in zip(examples, predictions, strict=True):
        lines.append(
            "| "
            f"`{example['id']}` | `{example['language']}` | "
            f"`{result['sentiment_label']}` | `{result['sentiment_confidence']:.4f}` | "
            f"`{result['authenticity_label']}` | `{result['authenticity_confidence']:.4f}` |"
        )

    for example, result in zip(examples, predictions, strict=True):
        explanation = result["explanation"]
        lines.append("")
        lines.append(f"## {example['title_ru']}")
        lines.append("")
        lines.append(f"- id: `{example['id']}`")
        lines.append(f"- why_in_demo: {example['why_in_demo']}")
        lines.append(f"- speaker_note: {example['speaker_note']}")
        lines.append(f"- text: {example['text']}")
        lines.append(
            f"- sentiment: `{result['sentiment_label']}` (`{result['sentiment_confidence']:.4f}`)"
        )
        lines.append(
            f"- authenticity: `{result['authenticity_label']}` (`{result['authenticity_confidence']:.4f}`)"
        )
        lines.append(
            "- sentiment_top_probabilities: "
            f"{probability_summary(explanation['sentiment_top_probabilities'])}"
        )
        lines.append(
            "- authenticity_top_probabilities: "
            f"{probability_summary(explanation['authenticity_top_probabilities'])}"
        )
        lines.append(
            f"- token_info: `{explanation['token_count']}/{explanation['max_length']}`"
            f", truncated=`{explanation['truncated']}`"
        )

    lines.append("")
    lines.append("## Presenter reminder")
    lines.append("")
    lines.append(
        "- This demo uses the current `pilot1k` multitask checkpoint, so it is a real model artifact rather than a mocked service."
    )
    lines.append(
        "- The strongest practical talking point is authenticity detection on exaggerated or AI-like reviews."
    )
    lines.append(
        "- The most honest limitation to show is weaker sentiment stability on Russian near-neutral product reviews."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a defense-ready demo report from the active ReviewGuard checkpoint."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/demo/reviews.json"),
        help="Path to the demo examples JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/defense_demo_results_ru.md"),
        help="Path to the markdown report that will be written.",
    )
    args = parser.parse_args()

    client = TestClient(app)
    health_response = client.get("/health")
    health_response.raise_for_status()
    health = health_response.json()
    if not health.get("model_ready"):
        raise SystemExit("Active checkpoint is not ready. Activate a multitask export first.")

    examples = load_examples(args.examples)
    predictions: list[dict[str, Any]] = []
    for example in examples:
        response = client.post("/analyze", json={"text": example["text"]})
        response.raise_for_status()
        predictions.append(response.json())

    report = render_markdown(examples, health, predictions)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
