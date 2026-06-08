# Pilot Results

## Real data prepared locally

- `RuReviews`: `60,602` normalized sentiment-labeled records
- `Perekrestok Reviews`: `642,682` normalized sentiment-labeled records
- `MAiDE-up`: `19,985` normalized records with sentiment and authenticity labels

## Pilot corpus

- `rureviews.pilot1k.jsonl`: `1,000` records
- `maide_up.pilot1k.jsonl`: `1,000` records
- `joint_reviews.pilot1k.jsonl`: `2,000` records total
- sampling seed: `42`
- split policy used by training CLI: `80/10/10`

## Baseline metrics

Source artifact: `models/pilot1k-baseline/train_report.json` if preserved separately during local experimentation. The artifact is not committed in the current repository snapshot.

### Test split

| Task | Accuracy | Macro-F1 | Weighted-F1 | Support |
|---|---:|---:|---:|---:|
| Sentiment | `0.8000` | `0.7368` | `0.8002` | `200` |
| Authenticity | `0.8000` | `0.7999` | `0.7999` | `100` |

### Validation split

| Task | Accuracy | Macro-F1 | Weighted-F1 | Support |
|---|---:|---:|---:|---:|
| Sentiment | `0.8200` | `0.8763` | `0.8200` | `200` |
| Authenticity | `0.8000` | `0.7999` | `0.7999` | `100` |

## Current blocker

The next step is the `single-task Transformer` and `multitask Transformer` comparison on the same pilot protocol. The local Python 3.14 environment is ready and the public datasets are already downloaded, but the first escalated Hugging Face model-download/training step was blocked by the platform usage limit rather than by repository code.

## Dissertation note

These pilot metrics are only a preliminary sanity check for the baseline pipeline on a small sampled corpus. They should not be presented as final dissertation results. The final manuscript should clearly separate:

- pilot validation on sampled data;
- full benchmark experiments on the complete corpora;
- comparative results for `baseline`, `single-task Transformer`, and `multitask Transformer`.
