# Workflow Guide

## Goal

The intended project workflow is:

1. place raw real datasets into `data/raw/...`
2. normalize them into one shared multitask schema
3. train baselines, single-task Transformers, and the multitask model
4. export the chosen checkpoint
5. run the API and web UI against a multitask checkpoint

## Unified schema

Each processed record should contain:

- `text`
- `sentiment_label`
- `authenticity_label`
- `source`
- `language`
- `domain`
- `metadata`

Missing labels are allowed. This is important because most public datasets only annotate one of the two tasks.

## Data normalization

CLI:

```bash
PYTHONPATH=src python3.14 -m reviewguard.data \
  --dataset rureviews \
  --input data/raw/rureviews/rureviews.csv \
  --output data/processed/rureviews.jsonl \
  --format jsonl
```

Supported adapters:

- `rureviews`
- `perekrestok`
- `opspam`
- `maide_up`

To build a shared multitask corpus from several processed files:

```bash
PYTHONPATH=src python3.14 -m reviewguard.data merge \
  --inputs data/processed/rureviews.jsonl data/processed/opspam.jsonl data/processed/maide_up.jsonl \
  --output data/processed/joint_reviews.jsonl \
  --format jsonl
```

## Baseline training

CLI:

```bash
PYTHONPATH=src python3.14 -m reviewguard.training baseline \
  --input data/processed/rureviews.jsonl \
  --export-dir models/baseline-rureviews
```

What it does:

- loads unified records
- creates `train/valid/test` splits
- fits `TF-IDF + LogisticRegression` per task when labels exist
- writes exported artifacts and `train_report.json`

## Multitask training

CLI:

```bash
PYTHONPATH=src python3.14 -m reviewguard.training multitask \
  --input data/processed/joint_reviews.jsonl \
  --export-dir models/latest \
  --config configs/model.multitask.yaml
```

What it does:

- loads unified records with partial labels
- trains one shared Transformer encoder with two classifier heads
- masks missing labels per task
- evaluates on validation and test splits
- exports a checkpoint consumable by the API

## Single-task Transformer training

CLI:

```bash
PYTHONPATH=src python3.14 -m reviewguard.training single-task \
  --input data/processed/joint_reviews.jsonl \
  --export-dir models/single-task-sentiment \
  --task sentiment \
  --config configs/model.multitask.yaml
```

```bash
PYTHONPATH=src python3.14 -m reviewguard.training single-task \
  --input data/processed/joint_reviews.jsonl \
  --export-dir models/single-task-authenticity \
  --task authenticity \
  --config configs/model.multitask.yaml
```

What it does:

- loads unified records and keeps the shared `train/valid/test` split logic
- filters each split to the requested task's labeled examples for fitting and scoring
- trains one Hugging Face sequence-classification model per task run
- writes a task-specific export with Hugging Face model files, `metadata.json`, `manifest.json`, and `train_report.json`

This is the clean comparison point for the dissertation before joint multitask training.
Single-task exports are not API-ready inference bundles for `/analyze`; the current service expects the multitask export format with `model.pt`.

## Export format

Multitask export directory:

```text
models/latest/
  encoder/
  metadata.json
  manifest.json
  model.pt
  tokenizer.json / tokenizer_config.json / special_tokens_map.json ...
```

## Serving

After exporting the multitask checkpoint:

```bash
uvicorn reviewguard.api.main:app --reload
```

Health endpoint:

- `GET /health`

Prediction endpoint:

- `POST /analyze`

The response now includes an `explanation` object with ranked task probabilities, token count, truncation status, and transparent notes.

## Recommended experiment order

1. `RuReviews` baseline for sentiment
2. `OpSpam` baseline for authenticity
3. merged partially labeled multitask corpus
4. `RuReviews + Perekrestok` for sentiment transfer
5. `FraudYelp + MAiDE-up` for authenticity robustness checks
