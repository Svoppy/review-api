import importlib
import sys

import pytest

from reviewguard.training.metrics import (
    compute_classification_metrics,
    compute_multitask_metrics,
    compute_task_metrics,
)
from reviewguard.training.splits import split_unified_records


def test_compute_classification_metrics_returns_expected_summary() -> None:
    metrics = compute_classification_metrics(
        ["negative", "neutral", "positive", "positive"],
        ["negative", "neutral", "negative", "positive"],
        labels=["negative", "neutral", "positive"],
    )

    assert metrics.support == 4
    assert metrics.accuracy == 0.75
    assert round(metrics.macro_f1, 4) == 0.7778
    assert metrics.per_label_support == {"negative": 1, "neutral": 1, "positive": 2}
    assert metrics.confusion_matrix == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_compute_multitask_metrics_ignores_missing_labels() -> None:
    records = [
        {
            "text": "good",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "bad",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": None,
            "authenticity_label": "fake",
        },
        {
            "text": "ok",
            "source": "b",
            "language": "ru",
            "domain": "ecommerce",
            "sentiment_label": "neutral",
            "authenticity_label": "authentic",
        },
    ]

    metrics = compute_multitask_metrics(
        records,
        sentiment_predictions=["positive", "negative", "neutral"],
        authenticity_predictions=["fake", "fake", "authentic"],
        sentiment_labels=["negative", "neutral", "positive"],
        authenticity_labels=["authentic", "fake"],
    )

    assert metrics["sentiment"]["support"] == 2
    assert metrics["sentiment"]["accuracy"] == 1.0
    assert metrics["authenticity"]["support"] == 2
    assert metrics["authenticity"]["accuracy"] == 1.0


def test_compute_task_metrics_returns_only_requested_task() -> None:
    records = [
        {
            "text": "great",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "positive",
            "authenticity_label": None,
        },
        {
            "text": "fine",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "neutral",
            "authenticity_label": "authentic",
        },
    ]

    metrics = compute_task_metrics(
        records,
        predictions=["positive", "negative"],
        task="sentiment",
        labels=["negative", "neutral", "positive"],
    )

    assert set(metrics.keys()) == {"sentiment"}
    assert metrics["sentiment"]["support"] == 2
    assert metrics["sentiment"]["accuracy"] == 0.5


def test_compute_task_metrics_returns_empty_when_task_has_no_labels() -> None:
    records = [
        {
            "text": "great",
            "source": "a",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": None,
            "authenticity_label": "authentic",
        }
    ]

    metrics = compute_task_metrics(
        records,
        predictions=["positive"],
        task="sentiment",
        labels=["negative", "neutral", "positive"],
    )

    assert metrics == {}


def test_build_arg_parser_supports_single_task_command() -> None:
    build_arg_parser = importlib.import_module("reviewguard.training.__main__").build_arg_parser
    args = build_arg_parser().parse_args(
        [
            "single-task",
            "--input",
            "data/unified.jsonl",
            "--export-dir",
            "models/sentiment-baseline",
            "--task",
            "sentiment",
        ]
    )

    assert args.command == "single-task"
    assert args.task == "sentiment"
    assert args.config_path == "configs/model.multitask.yaml"
    assert args.random_state is None


def test_single_task_config_prefers_cli_random_state_over_yaml() -> None:
    module = importlib.import_module("reviewguard.training.__main__")
    config = module._resolve_single_task_config(
        {
            "train": {"random_state": 7},
            "single_task": {
                "train": {"random_state": 11},
                "tasks": {"sentiment": {"random_state": 13}},
            },
        },
        "sentiment",
        random_state=23,
    )

    assert config.random_state == 23


def test_single_task_config_uses_yaml_random_state_when_cli_missing() -> None:
    module = importlib.import_module("reviewguard.training.__main__")
    config = module._resolve_single_task_config(
        {
            "train": {"random_state": 7},
            "single_task": {
                "train": {"random_state": 11},
                "tasks": {"sentiment": {"random_state": 13}},
            },
        },
        "sentiment",
        random_state=None,
    )

    assert config.random_state == 13


def test_split_random_state_uses_config_when_cli_missing() -> None:
    module = importlib.import_module("reviewguard.training.__main__")
    split_seed = module._resolve_split_random_state(
        None,
        {
            "train": {"random_state": 7},
            "single_task": {
                "train": {"random_state": 11},
                "tasks": {"sentiment": {"random_state": 13}},
            },
        },
        task="sentiment",
    )

    assert split_seed == 13


def test_split_unified_records_preserves_all_examples() -> None:
    records = [
        {
            "text": f"review-{index}",
            "source": "demo",
            "language": "en",
            "domain": "ecommerce",
            "sentiment_label": "positive" if index % 2 == 0 else "negative",
            "authenticity_label": "authentic" if index % 2 == 0 else "fake",
        }
        for index in range(10)
    ]

    split = split_unified_records(
        records,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
        random_state=7,
    )

    assert len(split["train"]) == 6
    assert len(split["valid"]) == 2
    assert len(split["test"]) == 2
    assert len(split["train"]) + len(split["valid"]) + len(split["test"]) == len(records)


def test_training_cli_module_import_does_not_require_torch() -> None:
    sys.modules.pop("reviewguard.training.__main__", None)
    module = importlib.import_module("reviewguard.training.__main__")
    assert module.TASK_CHOICES == ("sentiment", "authenticity")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="optional training dependencies are not installed",
)
def test_single_task_trainer_seeds_before_model_initialization(monkeypatch) -> None:
    import reviewguard.training.single_task as single_task

    events: list[tuple[str, object]] = []

    def fake_manual_seed(seed: int) -> None:
        events.append(("manual_seed", seed))

    monkeypatch.setattr(single_task.torch, "manual_seed", fake_manual_seed)
    monkeypatch.setattr(single_task.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        single_task.AutoTokenizer,
        "from_pretrained",
        lambda model_name: object(),
    )
    monkeypatch.setattr(
        single_task.AutoConfig,
        "from_pretrained",
        lambda model_name, num_labels: type("Cfg", (), {"num_labels": num_labels})(),
    )

    class DummyModel:
        def to(self, device):
            events.append(("model_to", str(device)))

    def fake_model_from_pretrained(model_name, config):
        events.append(("model_init", model_name))
        return DummyModel()

    monkeypatch.setattr(
        single_task.AutoModelForSequenceClassification,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    trainer = single_task.SingleTaskTransformerTrainer(
        single_task.SingleTaskTrainingConfig(
            task="sentiment",
            labels=["negative", "neutral", "positive"],
            random_state=17,
        )
    )

    assert trainer.task == "sentiment"
    assert events[0] == ("manual_seed", 17)
    assert events[1] == ("model_init", trainer.config.model_name)
