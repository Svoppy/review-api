# Data Acquisition Notes

This folder is intentionally empty in git. Real datasets should be downloaded or prepared locally.

## Phase-1 target datasets

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
