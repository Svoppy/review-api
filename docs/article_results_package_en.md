# Article Results Package

This file is generated from the current report artifacts and serves as the canonical article-facing appendix for the current repository snapshot.

## Evidence Boundary

- Tables 2-4 below are based on the **completed legacy pilot package** `pilot1k` and should be treated as the current executed comparative evidence layer.
- The strengthened rerun `pilot1k_v2` is already partially present in the repository, but it is not yet complete across all multitask seeds and does not yet have a final aggregated summary.
- Tables 5-6 are therefore readiness artifacts for the stronger rerun path, not new balanced6k model evidence.
- Because of that, `pilot1k_v2` should not yet be read as a replacement for the legacy article results; its status is tracked separately in [pilot1k_v2_status_en.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/pilot1k_v2_status_en.md).

## Sources

- pilot summary: `reports/multiseed/pilot1k/summary.json`
- statistics summary: `reports/multiseed/pilot1k_statistics/statistics_summary.json`
- balanced audit: `reports/audit/joint_reviews.balanced6k.audit.json`

## Table 1. Dataset Snapshot, Local Status, and Evidence Status

| Dataset | Language | Domain | Local Prepared Size | Label Type | Task | Evidence Status | Current Limitation |
|---|---|---|---:|---|---|---|---|
| `RuReviews` | RU | e-commerce | 60602 | direct sentiment labels | Sentiment | prepared locally; executed in legacy pilot | no authenticity label coverage |
| `Perekrestok Reviews` | RU | retail | 642682 | rating-derived sentiment | Sentiment | prepared locally; not executed in reported pilot | no authenticity labels; sentiment is partially heuristic |
| `OpSpam` | EN | hospitality | not prepared locally | truthful/deceptive labels | Authenticity | adapter planned; not prepared locally | planned extension; raw text not prepared locally |
| `FraudDataset (Yelp)` | EN | local commerce | not prepared locally | silver fraud labels | Authenticity | adapter ready; local text export missing | adapter ready; local text export not prepared |
| `MAiDE-up` | multilingual | hospitality | 19985 | sentiment + AI-generated authenticity | Sentiment + Authenticity | prepared locally; executed in legacy pilot | current authenticity evidence still comes only from this family |

## Part A. Executed Legacy Evidence

The following tables are the completed and aggregated `pilot1k` model evidence.

## Table 2. Executed Low-Resource Model Comparison (`pilot1k` Legacy Package)

| Model | Corpus | Task | Accuracy | Macro-F1 | Weighted-F1 | Precision_macro | Recall_macro | Mean +/- Std | Significance Reading |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| TF-IDF + Logistic Regression | pilot1k | Sentiment | 0.8000 | 0.7368 | 0.8002 | 0.7385 | 0.7370 | 0.7368 +/- 0.0000 | reference classical baseline |
| Single-task Transformer | pilot1k | Sentiment | 0.8150 | 0.5814 | 0.8063 | 0.6024 | 0.5782 | 0.5814 +/- 0.0526 | reference Transformer comparison |
| Multitask Transformer | pilot1k | Sentiment | 0.7317 | 0.5856 | 0.7234 | 0.6163 | 0.5841 | 0.5856 +/- 0.0665 | tied with single-task, worse than baseline (p=0.9510; p=0.0025) |
| TF-IDF + Logistic Regression | pilot1k | Authenticity | 0.8000 | 0.7999 | 0.7999 | 0.8005 | 0.8000 | 0.7999 +/- 0.0000 | reference classical baseline |
| Single-task Transformer | pilot1k | Authenticity | 0.8400 | 0.8328 | 0.8328 | 0.8740 | 0.8400 | 0.8328 +/- 0.1075 | reference Transformer comparison |
| Multitask Transformer | pilot1k | Authenticity | 0.9167 | 0.9160 | 0.9160 | 0.9248 | 0.9167 | 0.9160 +/- 0.0524 | better than single-task and baseline (p=0.0005; p=0.0005) |

## Table 3. Paired Macro-F1 Comparisons on the Fixed Test Split

| Task | Model A | Model B | Delta | 95% CI | p-value |
|---|---|---|---:|---|---:|
| Authenticity | multitask | single-task-authenticity | 0.0832 | [0.0447, 0.1274] | 0.0005 |
| Authenticity | multitask | baseline | 0.1160 | [0.0447, 0.1988] | 0.0005 |
| Sentiment | multitask | single-task-sentiment | 0.0042 | [-0.0915, 0.0954] | 0.9510 |
| Sentiment | multitask | baseline | -0.1513 | [-0.2504, -0.0446] | 0.0025 |

## Table 4. Stability Across Train Seeds

| Model | Task | Mean Accuracy | Std Accuracy | Mean Macro-F1 | Std Macro-F1 |
|---|---|---:|---:|---:|---:|
| Single-task Transformer | Sentiment | 0.8150 | 0.0132 | 0.5814 | 0.0526 |
| Multitask Transformer | Sentiment | 0.7317 | 0.0355 | 0.5856 | 0.0665 |
| Single-task Transformer | Authenticity | 0.8400 | 0.0954 | 0.8328 | 0.1075 |
| Multitask Transformer | Authenticity | 0.9167 | 0.0513 | 0.9160 | 0.0524 |

## Part B. Strengthened Rerun Readiness Only

The following tables are readiness artifacts only. No `balanced6k` model reruns are reported in this appendix yet.

## Table 5. `balanced6k` Pre-Rerun Audit and Readiness Snapshot

- records: `6000`
- source distribution: `rureviews=2500`, `perekrestok=2500`, `maide_up=1000`
- split overlap: `train/valid normalized=0`, `train/test normalized=0`, `valid/test normalized=0`
- duplicate summary: `exact=81`, `normalized=97`
- authenticity source coverage: `maide_up=1.0`, `rureviews=0.0`, `perekrestok=0.0`

## Table 6. `balanced6k` Pre-Rerun Class Balance and Majority Baseline

| Corpus | Task | Class | Count | Share |
|---|---|---|---:|---:|
| `balanced6k` | Sentiment | `negative` | 187 | 0.3086 |
| `balanced6k` | Sentiment | `neutral` | 8 | 0.0132 |
| `balanced6k` | Sentiment | `positive` | 411 | 0.6782 |
| `balanced6k` | Authenticity | `authentic` | 51 | 0.5100 |
| `balanced6k` | Authenticity | `fake` | 49 | 0.4900 |

| Corpus | Task | Majority Baseline Accuracy | Majority Baseline Macro-F1 |
|---|---|---:|---:|
| `balanced6k` | Sentiment | 0.6782 | 0.2694 |
| `balanced6k` | Authenticity | 0.5100 | 0.3377 |

## Table 7. Representative Confusion Matrices on Pilot Checkpoints

### Baseline Sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 77 | 0 | 22 |
| `neutral` | 0 | 3 | 2 |
| `positive` | 14 | 2 | 80 |

### Single-task Sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 89 | 0 | 10 |
| `neutral` | 4 | 0 | 1 |
| `positive` | 24 | 0 | 72 |

### Multitask Sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 86 | 3 | 10 |
| `neutral` | 2 | 2 | 1 |
| `positive` | 43 | 1 | 52 |

### Multitask Authenticity

| gold ↓ / pred → | `authentic` | `fake` |
|---|---|---|
| `authentic` | 37 | 13 |
| `fake` | 1 | 49 |

### Single-task Authenticity

| gold ↓ / pred → | `authentic` | `fake` |
|---|---|---|
| `authentic` | 23 | 27 |
| `fake` | 0 | 50 |

## Brief Error Analysis

- the baseline recovers the rare `neutral` sentiment class in only `3/5` pilot examples;
- the single-task sentiment Transformer collapses on `neutral` with `0/5` correct;
- the multitask sentiment model still overpredicts `negative` for positive reviews (`positive -> negative = 43`);
- the multitask authenticity model misses only `1` fake review on the representative pilot checkpoint;
- the single-task authenticity model overpredicts `fake`, producing `27` `authentic -> fake` errors.

## What Is Already Citable as Executed Evidence

- the `baseline / single-task / multitask` pilot comparison is complete on one fixed split;
- the pilot package already includes `mean +/- std`, bootstrap confidence intervals, and approximate randomization tests;
- the confusion matrices already show the qualitative structure of pilot model errors.

## What Is Already in Place for the Stronger Rerun

- the `balanced6k` snapshot already passes a leakage-safe split audit with zero `normalized_text_overlap` across splits;
- provenance and label coverage are now explicit in the unified pipeline.

## What Still Prevents a Genuine 10/10 Rating

- authenticity coverage in the current expanded snapshot still comes only from `MAiDE-up`;
- the balanced6k sentiment test split still contains only `8` `neutral` examples;
- full article-grade robustness and ablation reruns under the stronger protocol are still pending.
