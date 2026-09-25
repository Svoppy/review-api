# Multi-Seed Statistical Analysis

- Input: `data/processed/joint_reviews.article30k.jsonl`
- Fixed split seed: `42`
- Training seeds: `11, 21, 42, 84, 126`

## Confidence intervals across train seeds

| Model | Split | Task | Metric | Mean | Std | 95% CI |
|---|---|---|---|---:|---:|---:|
| `baseline` | `test` | `authenticity` | `accuracy` | 0.9000 | 0.0000 | [0.9000, 0.9000] |
| `baseline` | `test` | `authenticity` | `macro_f1` | 0.8998 | 0.0000 | [0.8998, 0.8998] |
| `baseline` | `test` | `sentiment` | `accuracy` | 0.8320 | 0.0000 | [0.8320, 0.8320] |
| `baseline` | `test` | `sentiment` | `macro_f1` | 0.7161 | 0.0000 | [0.7161, 0.7161] |
| `multitask` | `test` | `authenticity` | `accuracy` | 0.9526 | 0.0094 | [0.9409, 0.9642] |
| `multitask` | `test` | `authenticity` | `macro_f1` | 0.9526 | 0.0094 | [0.9409, 0.9642] |
| `multitask` | `test` | `sentiment` | `accuracy` | 0.8950 | 0.0047 | [0.8892, 0.9009] |
| `multitask` | `test` | `sentiment` | `macro_f1` | 0.7709 | 0.0087 | [0.7600, 0.7817] |
| `single-task-authenticity` | `test` | `authenticity` | `accuracy` | 0.9571 | 0.0087 | [0.9463, 0.9680] |
| `single-task-authenticity` | `test` | `authenticity` | `macro_f1` | 0.9571 | 0.0088 | [0.9463, 0.9680] |
| `single-task-sentiment` | `test` | `sentiment` | `accuracy` | 0.8958 | 0.0038 | [0.8911, 0.9004] |
| `single-task-sentiment` | `test` | `sentiment` | `macro_f1` | 0.7721 | 0.0087 | [0.7613, 0.7830] |

## Paired model comparisons on the fixed test split

| Task | Metric | A | B | Mean Delta | 95% Bootstrap CI | Approx. Randomization p | Cohen's dz |
|---|---|---|---|---:|---:|---:|---:|
| `authenticity` | `accuracy` | `multitask` | `single-task-authenticity` | -0.0046 | [-0.0109, 0.0020] | 0.1769 | -0.2727 |
| `authenticity` | `macro_f1` | `multitask` | `single-task-authenticity` | -0.0046 | [-0.0109, 0.0020] | 0.1829 | -0.2724 |
| `authenticity` | `accuracy` | `multitask` | `baseline` | 0.0526 | [0.0303, 0.0749] | 0.0005 | 5.5989 |
| `authenticity` | `macro_f1` | `multitask` | `baseline` | 0.0528 | [0.0305, 0.0749] | 0.0005 | 5.6106 |
| `sentiment` | `accuracy` | `multitask` | `single-task-sentiment` | -0.0007 | [-0.0047, 0.0034] | 0.7571 | -0.1147 |
| `sentiment` | `macro_f1` | `multitask` | `single-task-sentiment` | -0.0013 | [-0.0105, 0.0081] | 0.8016 | -0.1383 |
| `sentiment` | `accuracy` | `multitask` | `baseline` | 0.0630 | [0.0510, 0.0746] | 0.0005 | 13.3975 |
| `sentiment` | `macro_f1` | `multitask` | `baseline` | 0.0548 | [0.0337, 0.0732] | 0.0005 | 6.2744 |

## Notes

- Confidence intervals for model-level metrics are Student-t 95% intervals over train seeds.
- Pairwise deltas use paired bootstrap resampling over the fixed test examples, averaged across train seeds.
- p-values come from an approximate randomization test on the same fixed test split.
- Cohen's dz is reported over paired train-seed deltas as a compact effect-size summary; with few seeds it should be treated as descriptive rather than definitive.
- This layer is meant to prevent overclaiming, not to overstate certainty.

## Statistical cautions


