# Single-Task vs Multitask Ablation

This report isolates the core architectural question of the pilot: how the shared multitask encoder behaves relative to the task-specific Transformer baselines and the classical baseline on each task.

## Summary Table

| Task | Baseline Macro-F1 | Single-task Macro-F1 | Multitask Macro-F1 | Multitask vs Single-task | Multitask vs Baseline |
|---|---:|---:|---:|---|---|
| `Sentiment` | 0.7368 +/- 0.0000 | 0.5814 +/- 0.0526 | 0.5856 +/- 0.0665 | no clear difference (p=0.9510, 95% CI [-0.0915, 0.0954]) | significant drop (p=0.0025, 95% CI [-0.2504, -0.0446]) |
| `Authenticity` | 0.7999 +/- 0.0000 | 0.8328 +/- 0.1075 | 0.9160 +/- 0.0524 | significant gain (p=0.0005, 95% CI [0.0447, 0.1274]) | significant gain (p=0.0005, 95% CI [0.0447, 0.1988]) |

## Interpretation

### Sentiment

- Baseline macro-F1: `0.7368 +/- 0.0000`; single-task macro-F1: `0.5814 +/- 0.0526`; multitask macro-F1: `0.5856 +/- 0.0665`.
- Multitask vs single-task: `+0.0042` macro-F1; no clear difference (p=0.9510, 95% CI [-0.0915, 0.0954]).
- Multitask vs baseline: `-0.1513` macro-F1; significant drop (p=0.0025, 95% CI [-0.2504, -0.0446]).
- The current pilot does not support a positive-transfer interpretation for sentiment: multitask learning is roughly tied with the single-task Transformer on macro-F1 but remains clearly below the classical baseline.

### Authenticity

- Baseline macro-F1: `0.7999 +/- 0.0000`; single-task macro-F1: `0.8328 +/- 0.1075`; multitask macro-F1: `0.9160 +/- 0.0524`.
- Multitask vs single-task: `+0.0832` macro-F1; significant gain (p=0.0005, 95% CI [0.0447, 0.1274]).
- Multitask vs baseline: `+0.1160` macro-F1; significant gain (p=0.0005, 95% CI [0.0447, 0.1988]).
- The current pilot supports a positive-transfer interpretation for authenticity: the shared encoder improves the trust-oriented task relative to both comparison families.

## Article-ready takeaway

- The pilot ablation supports asymmetric transfer rather than uniform multitask gains.
- For authenticity, shared representations appear beneficial on the current low-resource mixed corpus.
- For sentiment, shared training does not beat the single-task Transformer and still underperforms the baseline, which means the current pilot cannot claim broad multitask superiority.

