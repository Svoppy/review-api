# Single-Task vs Multitask Ablation

This report isolates the core architectural question of the pilot: how the shared multitask encoder behaves relative to the task-specific Transformer baselines and the classical baseline on each task.

## Summary Table

| Task | Baseline Macro-F1 | Single-task Macro-F1 | Multitask Macro-F1 | Multitask vs Single-task | Multitask vs Baseline |
|---|---:|---:|---:|---|---|
| `Sentiment` | 0.7161 +/- 0.0000 | 0.7721 +/- 0.0087 | 0.7709 +/- 0.0087 | no clear difference (p=0.8016, 95% CI [-0.0105, 0.0081]) | significant gain (p=0.0005, 95% CI [0.0337, 0.0732]) |
| `Authenticity` | 0.8998 +/- 0.0000 | 0.9571 +/- 0.0088 | 0.9526 +/- 0.0094 | no clear difference (p=0.1829, 95% CI [-0.0109, 0.0020]) | significant gain (p=0.0005, 95% CI [0.0305, 0.0749]) |

## Interpretation

### Sentiment

- Baseline macro-F1: `0.7161 +/- 0.0000`; single-task macro-F1: `0.7721 +/- 0.0087`; multitask macro-F1: `0.7709 +/- 0.0087`.
- Multitask vs single-task: `-0.0013` macro-F1; no clear difference (p=0.8016, 95% CI [-0.0105, 0.0081]).
- Multitask vs baseline: `+0.0548` macro-F1; significant gain (p=0.0005, 95% CI [0.0337, 0.0732]).
- The current pilot does not support a positive-transfer interpretation for sentiment: multitask learning is roughly tied with the single-task Transformer on macro-F1 but remains clearly below the classical baseline.

### Authenticity

- Baseline macro-F1: `0.8998 +/- 0.0000`; single-task macro-F1: `0.9571 +/- 0.0088`; multitask macro-F1: `0.9526 +/- 0.0094`.
- Multitask vs single-task: `-0.0046` macro-F1; no clear difference (p=0.1829, 95% CI [-0.0109, 0.0020]).
- Multitask vs baseline: `+0.0528` macro-F1; significant gain (p=0.0005, 95% CI [0.0305, 0.0749]).
- The current pilot supports a positive-transfer interpretation for authenticity: the shared encoder improves the trust-oriented task relative to both comparison families.

## Article-ready takeaway

- The pilot ablation supports asymmetric transfer rather than uniform multitask gains.
- For authenticity, shared representations appear beneficial on the current low-resource mixed corpus.
- For sentiment, shared training does not beat the single-task Transformer and still underperforms the baseline, which means the current pilot cannot claim broad multitask superiority.

