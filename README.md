# ReviewGuard

`ReviewGuard` is a starter implementation for the dissertation topic:

`Development of a multitask transformer-based model and web system for joint sentiment analysis and authenticity of reviews in e-commerce`.

## What is included now

- a clean project structure for ML, API, web UI, and experiment configs
- a raw-to-unified data pipeline for real review datasets
- a classical baseline flow, a single-task Transformer baseline, and a multitask training scaffold
- a multitask Transformer model with two heads:
  - `sentiment`: `negative`, `neutral`, `positive`
  - `authenticity`: `authentic`, `fake`
- a `FastAPI` backend with an `/analyze` endpoint
- a simple browser UI for testing inference
- research notes with current, real datasets and their caveats
- reproducible dataset download/export and pilot-sampling scripts

## Current status

The full workflow is now scaffolded end to end:

1. normalize raw datasets into one multitask schema
2. split records into `train/valid/test`
3. train either a classical baseline, a single-task Transformer, or the multitask Transformer
4. export a checkpoint directory
5. serve a multitask exported checkpoint through the API

There is still no trained checkpoint committed in the repository. Until you train and export a multitask checkpoint into `models/latest`, `/analyze` will return a transparent error message instead of pretending to predict.

## Recommended first stack

- model backbone: `FacebookAI/xlm-roberta-base`
- training: `PyTorch + Hugging Face Transformers`
- backend: `FastAPI`
- web UI: plain `HTML/CSS/JS`

`XLM-R` is a practical starting point because it is multilingual and supports Russian and English review text in one encoder.

## Recommended real datasets for phase 1

- `RuReviews` as the primary Russian e-commerce sentiment benchmark
- `Perekrestok Reviews` as a larger in-domain retail corpus
- `FraudYelpDataset` as the main authenticity benchmark with real-world fraud signals
- `OpSpam` as the clean text-only authenticity benchmark
- `MAiDE-up` as the modern AI-generated fake-review benchmark

## Project layout

```text
configs/
data/
docs/
src/reviewguard/
  api/
  data/
  ml/
  training/
tests/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
uvicorn reviewguard.api.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Normalize raw data

To download the public Hugging Face datasets used in the project:

```bash
PYTHONPATH=src .venv314/bin/python scripts/download_public_datasets.py \
  --datasets perekrestok maide_up \
  --output-root data/raw
```

To create a deterministic pilot subset for CPU-friendly experiments:

```bash
PYTHONPATH=src .venv314/bin/python scripts/sample_unified_dataset.py \
  --input data/processed/rureviews.jsonl \
  --output data/processed/rureviews.pilot1k.jsonl \
  --max-rows 1000 \
  --seed 42 \
  --stratify-field sentiment_label
```

Example for `RuReviews`:

```bash
PYTHONPATH=src python3.14 -m reviewguard.data \
  --dataset rureviews \
  --input data/raw/rureviews/rureviews.csv \
  --output data/processed/rureviews.jsonl \
  --format jsonl
```

Supported dataset adapters right now:

- `rureviews`
- `perekrestok`
- `opspam`
- `maide_up`

To merge several processed sources into one joint corpus:

```bash
PYTHONPATH=src python3.14 -m reviewguard.data merge \
  --inputs data/processed/rureviews.jsonl data/processed/opspam.jsonl \
  --output data/processed/joint_reviews.jsonl \
  --format jsonl
```

## Train a classical baseline

```bash
PYTHONPATH=src python3.14 -m reviewguard.training baseline \
  --input data/processed/rureviews.jsonl \
  --export-dir models/baseline-rureviews
```

This writes a `manifest.json`, task model files, and `train_report.json`.

## Train a single-task Transformer baseline

```bash
PYTHONPATH=src python3.14 -m reviewguard.training single-task \
  --task sentiment \
  --input data/processed/rureviews.jsonl \
  --export-dir models/single-task-sentiment \
  --config configs/model.multitask.yaml
```

Use `--task authenticity` for the authenticity-only baseline.

Single-task exports are baseline artifacts for comparison and analysis. The current API loader does not consume them directly; `/analyze` expects a multitask export layout with `model.pt`.

## Train the multitask Transformer

```bash
PYTHONPATH=src python3.14 -m reviewguard.training multitask \
  --input data/processed/joint_reviews.jsonl \
  --export-dir models/latest \
  --config configs/model.multitask.yaml
```

The multitask export writes:

- `metadata.json`
- `manifest.json`
- `model.pt`
- `encoder/`
- tokenizer files in the export root

## Serve the trained model

Once `models/latest` contains an exported multitask checkpoint:

```bash
uvicorn reviewguard.api.main:app --reload
```

Then `POST /analyze` and the web UI will use the real model.

Example response shape:

```json
{
  "sentiment_label": "positive",
  "sentiment_confidence": 0.91,
  "authenticity_label": "authentic",
  "authenticity_confidence": 0.88,
  "model_name": "models/latest/encoder",
  "explanation": {
    "sentiment_top_probabilities": [
      { "label": "positive", "probability": 0.91 },
      { "label": "neutral", "probability": 0.07 },
      { "label": "negative", "probability": 0.02 }
    ],
    "authenticity_top_probabilities": [
      { "label": "authentic", "probability": 0.88 },
      { "label": "fake", "probability": 0.12 }
    ],
    "notes": [
      "Probabilities are ranked independently for sentiment and authenticity, and each task's list sums to 1.0.",
      "These scores show the model's relative preference inside each task and should not be treated as calibrated certainty.",
      "The full review fit within the model limit (87/256 tokens used)."
    ],
    "token_count": 87,
    "max_length": 256,
    "truncated": false
  }
}
```

## What is completed

1. current dataset research and selection
2. unified data schema and preprocessing CLI
3. classical baseline training and evaluation scaffold
4. single-task Transformer baseline scaffold
5. multitask training and export scaffold
6. API inference wiring with lightweight explainability
7. reproducible public-dataset export and pilot sampling utilities

## Next milestones for the dissertation itself

1. acquire the full real datasets locally
2. run experiments and collect metric tables
3. compare single-task vs multitask results
4. add interpretability and error analysis
5. prepare dissertation figures and methodology text

## Current documents

- [Architecture notes](docs/architecture.md)
- [Dataset notes](docs/datasets.md)
- [Workflow guide](docs/workflow.md)
- [Roadmap](docs/roadmap.md)
