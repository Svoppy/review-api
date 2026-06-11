# Multi-Seed Experiment Summary

- Input: `data/processed/joint_reviews.pilot1k.jsonl`
- Fixed split seed: `42`
- Training seeds: `11, 21, 42`

## Test Macro-F1

| Model | Task | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `baseline` | `authenticity` | 0.7999 +/- 0.0000 | 0.8000 +/- 0.0000 |
| `baseline` | `sentiment` | 0.7368 +/- 0.0000 | 0.8000 +/- 0.0000 |
| `multitask` | `authenticity` | 0.9160 +/- 0.0524 | 0.9167 +/- 0.0513 |
| `multitask` | `sentiment` | 0.5856 +/- 0.0665 | 0.7317 +/- 0.0355 |
| `single-task-authenticity` | `authenticity` | 0.8328 +/- 0.1075 | 0.8400 +/- 0.0954 |
| `single-task-sentiment` | `sentiment` | 0.5814 +/- 0.0526 | 0.8150 +/- 0.0132 |

