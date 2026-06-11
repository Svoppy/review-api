# Data Acquisition Notes

This folder is intentionally empty in git. Real datasets should be downloaded or prepared locally.

## Current local state

The current workspace contains local raw copies only for:

- `RuReviews`
- `Perekrestok Reviews`
- `MAiDE-up`

There is no local `data/raw/opspam/` directory in this snapshot, and no local `FraudYelpDataset` review-text export has been prepared yet.

## Intended dissertation dataset stack

### Sentiment

- `RuReviews`: <https://github.com/sismetanin/rureviews>
- `Perekrestok Reviews`: <https://huggingface.co/datasets/lapki/perekrestok-reviews>
- `KazSAnDRA`: <https://github.com/IS2AI/KazSAnDRA>

### Authenticity

- `FraudYelpDataset`: <https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudDataset.html>
- `OpSpam`: <https://myleott.com/op-spam.html>
- `MAiDE-up`: <https://huggingface.co/datasets/MichiganNLP/MAiDE-up>

## Local convention

Store raw files under task-specific subfolders, for example:

```text
data/raw/rureviews/
data/raw/perekrestok/
data/raw/fraudyelp/
data/raw/opspam/
data/raw/maide_up/
```

Only the first three of those directories are currently present in this workspace.

After normalization, export unified records into:

```text
data/processed/
```

## Unified schema target

Each processed record should contain:

- `text`
- `sentiment_label`
- `authenticity_label`
- `language`
- `domain`
- `source`
- `metadata`

## FraudYelp local export contract

For this repository, `FraudYelpDataset` is treated as a local text-export integration rather than a raw graph loader.

Recommended local filenames:

```text
data/raw/fraudyelp/reviews.jsonl
data/raw/fraudyelp/reviews.csv
data/raw/fraudyelp/yelp_reviews.jsonl
```

Each row should already contain review text plus a fraud label field. The current loader does not claim to consume arbitrary upstream DGL graph artifacts directly.
