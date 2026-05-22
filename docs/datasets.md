# Dataset Notes

Verified on `2026-05-21`.

## Recommendation summary

There is no perfect modern public dataset that jointly labels both e-commerce sentiment and review authenticity at scale. The most defensible dissertation path is:

1. use a real large review corpus for sentiment
2. use established public authenticity benchmarks with explicit caveats
3. create a unified training format and, later, a small manually reviewed evaluation subset for joint analysis

## Sentiment datasets

### RuReviews

- source: <https://github.com/sismetanin/rureviews>
- type: Russian product reviews
- fit: the strongest current Russian benchmark for e-commerce sentiment
- labels: released as `negative`, `neutral`, `positive`
- note: this should be the main supervised sentiment benchmark in the dissertation

### Perekrestok Reviews

- source: <https://huggingface.co/datasets/lapki/perekrestok-reviews>
- type: Russian grocery and retail product reviews
- fit: strong in-domain corpus for additional pretraining, weak supervision, or domain adaptation
- labels: sentiment can be derived from the original rating
- note: a good large-scale companion to `RuReviews`

### KazSAnDRA

- sources: <https://github.com/IS2AI/KazSAnDRA>, <https://aclanthology.org/2024.lrec-main.844/>
- type: Kazakh, Russian, English, and code-switched app reviews
- fit: useful if we want a stronger regional multilingual contribution
- labels: polarity and score variants are available
- caveat: not e-commerce specific, but valuable for robustness experiments

### Amazon Reviews 2023

- source: <https://amazon-reviews-2023.github.io/main.html>
- type: real e-commerce reviews
- fit: strong for large-scale transfer learning and auxiliary experiments
- signal: derive sentiment from star ratings
- caveat: not Russian and too large to be the only practical dissertation dataset

## Authenticity datasets

### FraudYelpDataset

- source: <https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudDataset.html>
- type: benchmark fraud labels on Yelp review graph data
- fit: the best current public starting point for a more realistic fake-review benchmark
- note: good as the main authenticity benchmark in the experimental stack
- caveat: labels should be treated as silver or benchmark labels, not perfect ground truth

### Deceptive Opinion Spam Corpus v1.4

- sources: <https://myleott.com/op-spam.html>, <https://aclanthology.org/P11-1032/>
- type: binary authenticity labels
- fit: strongest classic text-only benchmark for deceptive review detection
- scope: hotel reviews
- caveat: deceptive reviews were crowdsourced in an experimental setting, so the distribution differs from live marketplace fraud

### MAiDE-up

- sources: <https://huggingface.co/datasets/MichiganNLP/MAiDE-up>, <https://arxiv.org/abs/2404.12938>
- type: real vs AI-generated fake reviews
- fit: highly relevant for a 2026 dissertation because LLM-generated manipulation is now part of the threat model
- caveat: this is AI-generated fraud, not the same phenomenon as human commercial review spam

### FraudAmazonDataset

- source: <https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudDataset.html>
- fit: useful supplementary benchmark for reviewer-level fraud analysis in the Amazon ecosystem
- caveat: labels are proxy labels, so this is better for fraud behavior experiments than pure review-text authenticity

## Recommended phase-1 data strategy

### For sentiment

Use `RuReviews` as the primary benchmark and `Perekrestok Reviews` as the main in-domain expansion corpus.

### For authenticity

Use two tracks:

- `FraudYelpDataset` for the main realistic benchmark
- `OpSpam` for the clean text-only benchmark
- `MAiDE-up` for the AI-generated fake-review setting

### For multitask training

Start with a shared schema:

- fields: `text`, `sentiment_label`, `authenticity_label`, `language`, `domain`, `source`
- allow missing labels per task

This lets us train a multitask model even when some examples are labeled for only one task.

## Recommended phase-1 stack

- core sentiment benchmark: `RuReviews`
- domain expansion: `Perekrestok Reviews`
- multilingual extension: `KazSAnDRA`
- main authenticity benchmark: `FraudYelpDataset`
- text-only authenticity benchmark: `OpSpam`
- AI-fake benchmark: `MAiDE-up`

## Scientific caution

For the dissertation text, we should state clearly that public authenticity datasets are heterogeneous:

- some contain crowd-generated deception
- some use platform filters or benchmark heuristics
- some focus on AI-generated reviews

That limitation is real and should be framed as part of the research problem, not hidden.
