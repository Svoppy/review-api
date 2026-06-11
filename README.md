# ReviewGuard

`ReviewGuard` is a starter implementation for the dissertation topic:

`Development of a multitask transformer-based model and web system for joint sentiment analysis and authenticity of reviews in e-commerce`.

The repository should currently be understood as a bounded empirical research package with a completed low-resource study, not yet as a completed multi-benchmark release.

Research stance: AI is a copilot here, not the pilot. The repository is designed to improve reproducibility, analysis quality, and deployment readiness, not to replace human methodological judgment.

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
- reproducible dataset download/export and study-sampling scripts

## Current status

The full workflow is now implemented end to end:

1. normalize raw datasets into one multitask schema
2. split records into `train/valid/test`
3. train either a classical baseline, a single-task Transformer, or the multitask Transformer
4. export a checkpoint directory
5. serve a multitask exported checkpoint through the API

The repository contains the full training, analysis, export, and API stack. However, the current scientific scope is still narrower than the intended dissertation benchmark:

- code-level support is broader than the locally prepared dataset bundle;
- the current reported evidence is a completed low-resource study package;
- broader benchmark claims still require larger and more source-controlled experiments.

Until you train and export a multitask checkpoint into `models/latest`, `/analyze` will return a transparent error message instead of pretending to predict.

## Recommended first stack

- model backbone: `FacebookAI/xlm-roberta-base`
- training: `PyTorch + Hugging Face Transformers`
- backend: `FastAPI`
- web UI: plain `HTML/CSS/JS`

`XLM-R` is a practical starting point because it is multilingual and supports Russian and English review text in one encoder.

Supported dataset loaders currently implemented in code:

- `rureviews`
- `perekrestok`
- `opspam`
- `fraudyelp`
- `maide_up`

Current local dataset preparation in this workspace is narrower than the code-level adapter list:

- `RuReviews`, `Perekrestok Reviews`, and `MAiDE-up` are prepared locally under `data/raw/` and `data/processed/`
- `OpSpam` and `FraudYelpDataset` have normalization adapters in code, but no local raw or processed copies are currently present in this workspace
- `FraudYelpDataset` currently expects a local review-text export; the repository does not pretend that an arbitrary raw graph dump is already text-ready by default

When describing completed experiments, only the datasets that are actually prepared locally and used in the reports should be presented as part of the current empirical scope.

## Current empirical claim

What the repository already supports:

- a reproducible low-resource comparison over `baseline`, `single-task`, and `multitask` models;
- repeated study runs across three train seeds;
- significance-oriented reporting and task-ablation artifacts;
- an API and browser UI that can consume an exported multitask checkpoint.

What remains explicitly human-in-the-loop:

- choosing the final research question and scope;
- judging whether label harmonization is scientifically acceptable;
- interpreting transfer effects and threats to validity;
- deciding which claims are strong enough to place in the article.

What it does not yet support as a final scientific claim:

- general multitask superiority across tasks;
- benchmark-level robustness claims across domains and languages;
- final authenticity conclusions across all planned benchmark families.

## Intended dissertation dataset stack

- `RuReviews` as the primary Russian e-commerce sentiment benchmark
- `Perekrestok Reviews` as a larger in-domain retail corpus
- `FraudYelpDataset` as the main authenticity benchmark with real-world fraud signals
- `OpSpam` as the clean text-only authenticity benchmark
- `MAiDE-up` as the modern AI-generated fake-review benchmark

## Current reported experiment scope

- locally prepared sentiment sources: `RuReviews`, `Perekrestok Reviews`
- locally prepared authenticity source: `MAiDE-up`
- current reported study scope: sampled subsets of `RuReviews` and `MAiDE-up`
- not part of the current reported benchmark: `OpSpam`, `FraudYelpDataset`

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

To expose an exported multitask model as the active API checkpoint:

```bash
PYTHONPATH=src python scripts/activate_checkpoint.py \
  --source models/pilot1k-multitask \
  --target models/latest
```

## Normalize raw data

To download the public Hugging Face datasets used in the project:

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets perekrestok maide_up --output-root data/raw
```

To create a deterministic study subset for CPU-friendly experiments:

```bash
PYTHONPATH=src python scripts/sample_unified_dataset.py \
  --input data/processed/rureviews.jsonl \
  --output data/processed/rureviews.pilot1k.jsonl \
  --max-rows 1000 \
  --seed 42 \
  --stratify-fields sentiment_label \
  --min-per-group 20
```

To create a larger, more reviewer-safe benchmark subset with composite stratification:

```bash
PYTHONPATH=src python scripts/sample_unified_dataset.py \
  --input data/processed/joint_reviews.current.jsonl \
  --output data/processed/joint_reviews.balanced6k.jsonl \
  --max-rows 6000 \
  --seed 42 \
  --stratify-fields source sentiment_label authenticity_label language domain \
  --min-per-group 40 \
  --source-cap rureviews=2500 \
  --source-cap perekrestok=2500 \
  --source-cap maide_up=1000
```

Example for `RuReviews`:

```bash
PYTHONPATH=src python -m reviewguard.data normalize --dataset rureviews --input data/raw/rureviews/rureviews.csv --output data/processed/rureviews.jsonl --format jsonl
```

Supported dataset adapters right now:

- `rureviews`
- `perekrestok`
- `opspam`
- `fraudyelp`
- `maide_up`

To merge the processed sources that are currently available in this workspace into one joint corpus:

```bash
PYTHONPATH=src python -m reviewguard.data merge --inputs data/processed/rureviews.jsonl data/processed/perekrestok.jsonl data/processed/maide_up.jsonl --output data/processed/joint_reviews.current.jsonl --format jsonl
```

If `OpSpam` is downloaded locally later, it can be merged as an extension corpus, but it should not be implied to exist in the current workspace snapshot by default.

To generate a dissertation-facing audit report with split sizes, class balance, duplicate-aware leakage checks, per-source label coverage, and majority baselines:

```bash
PYTHONPATH=src python -m reviewguard.data audit --input data/processed/joint_reviews.current.jsonl --output reports/joint_reviews.audit.json --random-state 42
```

## Train a classical baseline

```bash
PYTHONPATH=src python -m reviewguard.training baseline --input data/processed/rureviews.jsonl --export-dir models/baseline-rureviews
```

This writes a `manifest.json`, task model files, and `train_report.json`.

## Train a single-task Transformer baseline

```bash
PYTHONPATH=src python -m reviewguard.training single-task --task sentiment --input data/processed/rureviews.jsonl --export-dir models/single-task-sentiment --config configs/model.multitask.yaml
```

Use `--task authenticity` for the authenticity-only baseline.

Single-task exports are baseline artifacts for comparison and analysis. The current API loader does not consume them directly; `/analyze` expects a multitask export layout with `model.pt`.

## Train the multitask Transformer

```bash
PYTHONPATH=src python -m reviewguard.training multitask --input data/processed/joint_reviews.current.jsonl --export-dir models/latest --config configs/model.multitask.yaml
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

The repository already has pilot multitask artifacts and reporting utilities, and activating an exported checkpoint through `scripts/activate_checkpoint.py` makes the API immediately usable.

Quick API smoke check with the active checkpoint:

```bash
PYTHONPATH=src python scripts/api_smoke.py
```

Example response shape:

```json
{
  "sentiment_label": "positive",
  "sentiment_confidence": 0.91,
  "authenticity_label": "authentic",
  "authenticity_confidence": 0.88,
  "model_name": "FacebookAI/xlm-roberta-base",
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
    "risk_flags": [],
    "token_count": 87,
    "max_length": 256,
    "truncated": false,
    "sentiment_margin": 0.84,
    "authenticity_margin": 0.76
  },
  "research_context": {
    "scope": "bounded_low_resource_study",
    "training_records": 2000,
    "sentiment_labeled": 2000,
    "authenticity_labeled": 1000
  }
}
```

## What is completed

1. current dataset research and selection
2. unified data schema and preprocessing CLI
3. classical baseline training and evaluation scaffold
4. single-task Transformer baseline scaffold
5. multitask training and export scaffold
6. duplicate-aware split audit and robustness-analysis utilities
7. API inference wiring with lightweight transparency and research-context metadata
8. reproducible public-dataset export and study sampling utilities
9. real low-resource baseline, single-task, and multitask experiment artifacts

## What still needs to be completed for the final dissertation benchmark

1. larger benchmark subsets with stronger minority-class support
2. source-controlled robustness evaluation across `source`, `domain`, and `language`
3. local integration of the remaining planned authenticity benchmarks
4. dataset-specific authenticity tables in the final article
5. final audit appendix linking the exact dataset bundle to the reported article numbers

## Defense pack

The repository now also includes a defense-oriented package:

- [Defense demo script (RU)](docs/defense_demo_script_ru.md)
- [Defense demo results (RU)](docs/defense_demo_results_ru.md)
- [Defense Q&A (RU)](docs/defense_qa_ru.md)
- [Defense slides outline (RU)](docs/defense_slides_outline_ru.md)
- [Demo review set](data/demo/reviews.json)

To regenerate the live-demo report from the active checkpoint without starting an external server:

```bash
PYTHONPATH=src python scripts/generate_defense_demo.py
```

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
- [Pilot results](docs/pilot_results.md)
- [Article results package](docs/article_results_package_ru.md)
- [Pilot analysis (EN)](docs/pilot_analysis_en.md)
- [Dissertation audit (RU)](docs/dissertation_audit_ru.md)
- [Reviewer response (RU)](docs/reviewer_response_ru.md)
- [Submission checklist (RU)](docs/article_submission_checklist_ru.md)
- [Results table templates (RU)](docs/results_tables_template_ru.md)
- [Reproducibility note (RU)](docs/reproducibility_note_ru.md)
- [Tested environment (RU)](docs/tested_environment_ru.md)
- [Defense demo script (RU)](docs/defense_demo_script_ru.md)
- [Defense Q&A (RU)](docs/defense_qa_ru.md)
- [Defense slides outline (RU)](docs/defense_slides_outline_ru.md)
- [Final article draft (RU)](docs/article_final_ru.md)
- [Final article in LaTeX (EN)](docs/article_final_en.tex)
- [Results table templates (EN)](docs/results_tables_template_en.md)
