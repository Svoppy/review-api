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
- test supports used in reporting: `200` sentiment examples, `100` authenticity examples

## Comparison artifacts

- baseline: [models/pilot1k-baseline/train_report.json](/Users/diaskazikhanov/Desktop/aitu/nirm/models/pilot1k-baseline/train_report.json)
- single-task sentiment: [models/pilot1k-single-task-sentiment/train_report.json](/Users/diaskazikhanov/Desktop/aitu/nirm/models/pilot1k-single-task-sentiment/train_report.json)
- single-task authenticity: [models/pilot1k-single-task-authenticity/train_report.json](/Users/diaskazikhanov/Desktop/aitu/nirm/models/pilot1k-single-task-authenticity/train_report.json)
- multitask: [models/pilot1k-multitask/train_report.json](/Users/diaskazikhanov/Desktop/aitu/nirm/models/pilot1k-multitask/train_report.json)
- multi-seed summary: [reports/multiseed/pilot1k/summary.md](/Users/diaskazikhanov/Desktop/aitu/nirm/reports/multiseed/pilot1k/summary.md)
- statistical summary: [reports/multiseed/pilot1k_statistics/statistics_report.md](/Users/diaskazikhanov/Desktop/aitu/nirm/reports/multiseed/pilot1k_statistics/statistics_report.md)
- task ablation: [reports/multiseed/pilot1k_ablation/task_ablation.md](/Users/diaskazikhanov/Desktop/aitu/nirm/reports/multiseed/pilot1k_ablation/task_ablation.md)

## Test split comparison

| Model | Sentiment Accuracy | Sentiment Macro-F1 | Authenticity Accuracy | Authenticity Macro-F1 |
|---|---:|---:|---:|---:|
| `TF-IDF + Logistic Regression baseline` | `0.8000` | `0.7368` | `0.8000` | `0.7999` |
| `Single-task Transformer (sentiment)` | `0.8050` | `0.5428` | `-` | `-` |
| `Single-task Transformer (authenticity)` | `-` | `-` | `0.7300` | `0.7088` |
| `Multitask Transformer` | `0.7000` | `0.5885` | `0.8600` | `0.8580` |

## Interpretation

- On this executed low-resource study protocol, the `multitask Transformer` achieved the best authenticity result.
- On the same study protocol, the `baseline` remained stronger for sentiment than both Transformer variants.
- The `single-task sentiment` model slightly exceeded the baseline in accuracy, but performed worse in macro-F1 because the rare `neutral` class remained difficult.
- The `single-task authenticity` model underperformed both the baseline and the multitask model.
- A direct inference smoke test also succeeded on the exported multitask checkpoint, confirming that `models/pilot1k-multitask` is a valid API-consumable artifact.

## Error analysis

- The sentiment test split contains only `5` `neutral` examples, so macro-F1 is highly sensitive to minority-class errors.
- The `single-task sentiment` Transformer predicted `0` `neutral` reviews correctly on the test split, which explains why its macro-F1 is much worse than its accuracy suggests.
- The `baseline` handled `neutral` more reliably (`3/5` correct) than the Transformer variants.
- The `multitask` sentiment model made many more `positive -> negative` mistakes than the baseline (`43` vs `14`), which is the main reason for its sentiment drop.
- The `single-task authenticity` Transformer achieved perfect fake-review recall (`50/50`) but overpredicted the `fake` label, misclassifying `27` authentic reviews.
- The `multitask` model preserved near-perfect fake-review recall (`49/50`) while producing a more balanced authenticity decision boundary than the single-task authenticity model.

## Task ablation takeaway

- `Sentiment`: `multitask` is statistically indistinguishable from the `single-task sentiment` Transformer on macro-F1, but significantly worse than the `baseline`.
- `Authenticity`: `multitask` is significantly better than both the `single-task authenticity` Transformer and the `baseline`.
- The pilot therefore supports `asymmetric transfer`, not a universal gain from multitask learning.

## Statistical reading

- `Authenticity` is the strongest result in the package: multi-seed macro-F1 is `0.9160 +/- 0.0524`, the multitask-vs-single-task delta is `+0.0832`, the paired 95% bootstrap interval stays positive (`[0.0447, 0.1274]`), and approximate randomization gives `p=0.0005`.
- `Sentiment` is not a positive-transfer result: multitask-vs-single-task macro-F1 is only `+0.0042` with a wide interval crossing zero (`[-0.0915, 0.0954]`) and `p=0.9510`.
- `Sentiment` multitask-vs-baseline is an explicit negative finding: delta `-0.1513`, interval `[-0.2504, -0.0446]`, `p=0.0025`.
- Reviewer-facing evidence level: `authenticity` is `solid within this pilot`, while `sentiment` should be treated as `caution` because the neutral class support is only `5`.
- The current significance layer is useful for calibration, but it is still modest in scope: only `3` train seeds were used, and the paper should say these are planned contrasts rather than a fully confirmatory, multiplicity-controlled analysis.

## Dissertation note

These study metrics are real experimental evidence on a small sampled corpus, but they should not be presented as a benchmark-wide conclusion. The final manuscript should clearly separate:

- bounded low-resource validation on sampled data;
- full benchmark experiments on the complete corpora;
- broader comparative results for `baseline`, `single-task Transformer`, and `multitask Transformer`;
- broader claims from the currently executed study observations.

## Next empirical step

The next run should not reuse the `2,000`-row pilot as the main evidence package. The repository now supports a stronger protocol:

- build a larger composite-stratified subset from `RuReviews`, `Perekrestok Reviews`, and `MAiDE-up`;
- enforce minimum support for rare strata before training;
- run split-level audit before each experiment;
- train with `4` epochs, class balancing, source-balanced multitask sampling, and early stopping;
- report source/domain/language robustness slices in addition to aggregate scores.
