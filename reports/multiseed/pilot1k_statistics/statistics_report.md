# Multi-Seed Statistical Analysis

- Input: `data/processed/joint_reviews.pilot1k.jsonl`
- Fixed split seed: `42`
- Training seeds: `11, 21, 42`

## Confidence intervals across train seeds

| Model | Split | Task | Metric | Mean | Std | 95% CI |
|---|---|---|---|---:|---:|---:|
| `baseline` | `test` | `authenticity` | `accuracy` | 0.8000 | 0.0000 | [0.8000, 0.8000] |
| `baseline` | `test` | `authenticity` | `macro_f1` | 0.7999 | 0.0000 | [0.7999, 0.7999] |
| `baseline` | `test` | `sentiment` | `accuracy` | 0.8000 | 0.0000 | [0.8000, 0.8000] |
| `baseline` | `test` | `sentiment` | `macro_f1` | 0.7368 | 0.0000 | [0.7368, 0.7368] |
| `multitask` | `test` | `authenticity` | `accuracy` | 0.9167 | 0.0513 | [0.7892, 1.0000] |
| `multitask` | `test` | `authenticity` | `macro_f1` | 0.9160 | 0.0524 | [0.7857, 1.0000] |
| `multitask` | `test` | `sentiment` | `accuracy` | 0.7317 | 0.0355 | [0.6435, 0.8198] |
| `multitask` | `test` | `sentiment` | `macro_f1` | 0.5856 | 0.0665 | [0.4204, 0.7508] |
| `single-task-authenticity` | `test` | `authenticity` | `accuracy` | 0.8400 | 0.0954 | [0.6030, 1.0000] |
| `single-task-authenticity` | `test` | `authenticity` | `macro_f1` | 0.8328 | 0.1075 | [0.5657, 1.0000] |
| `single-task-sentiment` | `test` | `sentiment` | `accuracy` | 0.8150 | 0.0132 | [0.7821, 0.8479] |
| `single-task-sentiment` | `test` | `sentiment` | `macro_f1` | 0.5814 | 0.0526 | [0.4508, 0.7120] |

## Paired model comparisons on the fixed test split

| Task | Metric | A | B | Mean Delta | 95% Bootstrap CI | Approx. Randomization p |
|---|---|---|---|---:|---:|---:|
| `authenticity` | `accuracy` | `multitask` | `single-task-authenticity` | 0.0767 | [0.0400, 0.1167] | 0.0005 |
| `authenticity` | `macro_f1` | `multitask` | `single-task-authenticity` | 0.0832 | [0.0447, 0.1274] | 0.0005 |
| `authenticity` | `accuracy` | `multitask` | `baseline` | 0.1167 | [0.0433, 0.2000] | 0.0005 |
| `authenticity` | `macro_f1` | `multitask` | `baseline` | 0.1160 | [0.0447, 0.1988] | 0.0005 |
| `sentiment` | `accuracy` | `multitask` | `single-task-sentiment` | -0.0833 | [-0.1267, -0.0400] | 0.0005 |
| `sentiment` | `macro_f1` | `multitask` | `single-task-sentiment` | 0.0042 | [-0.0915, 0.0954] | 0.9510 |
| `sentiment` | `accuracy` | `multitask` | `baseline` | -0.0683 | [-0.1150, -0.0216] | 0.0010 |
| `sentiment` | `macro_f1` | `multitask` | `baseline` | -0.1513 | [-0.2504, -0.0446] | 0.0025 |

## Notes

- Confidence intervals for model-level metrics are Student-t 95% intervals over train seeds.
- Pairwise deltas use paired bootstrap resampling over the fixed test examples, averaged across train seeds.
- p-values come from an approximate randomization test on the same fixed test split.
- With only three train seeds, this layer is still preliminary; it is meant to prevent overclaiming, not to overstate certainty.

