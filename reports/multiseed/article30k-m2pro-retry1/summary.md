# Multi-Seed Experiment Summary

- Input: `data/processed/joint_reviews.article30k.jsonl`
- Fixed split seed: `42`
- Training seeds: `11, 21, 42, 84, 126`

## Test Macro-F1

| Model | Task | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `baseline` | `authenticity` | 0.8998 +/- 0.0000 | 0.9000 +/- 0.0000 |
| `baseline` | `sentiment` | 0.7161 +/- 0.0000 | 0.8320 +/- 0.0000 |
| `multitask` | `authenticity` | 0.9526 +/- 0.0094 | 0.9526 +/- 0.0094 |
| `multitask` | `sentiment` | 0.7709 +/- 0.0087 | 0.8950 +/- 0.0047 |
| `single-task-authenticity` | `authenticity` | 0.9571 +/- 0.0088 | 0.9571 +/- 0.0087 |
| `single-task-sentiment` | `sentiment` | 0.7721 +/- 0.0087 | 0.8958 +/- 0.0038 |

