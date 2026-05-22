# ReviewGuard

`ReviewGuard` is a starter implementation for the dissertation topic:

`Development of a multitask transformer-based model and web system for joint sentiment analysis and authenticity of reviews in e-commerce`.

## What is included now

- a clean project structure for ML, API, web UI, and experiment configs
- a multitask Transformer model with two heads:
  - `sentiment`: `negative`, `neutral`, `positive`
  - `authenticity`: `authentic`, `fake`
- a `FastAPI` backend with an `/analyze` endpoint
- a simple browser UI for testing inference
- research notes with current, real datasets and their caveats

## Important note

The API scaffold is ready, but there is no trained checkpoint in the repository yet. Until we train and save a model, `/analyze` will return a transparent error message instead of pretending to predict.

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
  ml/
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

## Next milestones

1. download and normalize real datasets
2. train single-task baselines
3. train the multitask model
4. compare metrics and error profiles
5. connect the best checkpoint to the web system

## Current documents

- [Architecture notes](docs/architecture.md)
- [Dataset notes](docs/datasets.md)
- [Roadmap](docs/roadmap.md)
