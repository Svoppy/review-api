# ReviewGuard

`ReviewGuard` is a starter implementation for the dissertation topic:

`Development of a multitask transformer-based model and web system for joint sentiment analysis and authenticity of reviews in e-commerce`.

## What is included now

- a clean project structure for ML, API, web UI, and experiment configs
- a raw-to-unified data pipeline for real review datasets
- a classical baseline training flow and a multitask training scaffold
- a multitask Transformer model with two heads:
  - `sentiment`: `negative`, `neutral`, `positive`
  - `authenticity`: `authentic`, `fake`
- a `FastAPI` backend with an `/analyze` endpoint
- a simple browser UI for testing inference
- research notes with current, real datasets and their caveats

## Current status

The full workflow is now scaffolded end to end:

1. normalize raw datasets into one multitask schema
2. split records into `train/valid/test`
3. train either a classical baseline or the multitask Transformer
4. export a checkpoint directory
5. serve that exported checkpoint through the API

There is still no trained checkpoint committed in the repository. Until you train and export one into `models/latest`, `/analyze` will return a transparent error message instead of pretending to predict.

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
pip install -e .
uvicorn reviewguard.api.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Normalize raw data

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

## Train a classical baseline

```bash
PYTHONPATH=src python3.14 -m reviewguard.training baseline \
  --input data/processed/rureviews.jsonl \
  --export-dir models/baseline-rureviews
```

This writes a `manifest.json`, task model files, and `train_report.json`.

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

## What is completed

1. current dataset research and selection
2. unified data schema and preprocessing CLI
3. baseline training and evaluation scaffold
4. multitask training and export scaffold
5. API inference wiring for exported checkpoints

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
