# Strengthened Pilot Rerun Status (`pilot1k_v2`)

Updated: `2026-06-12`

## What This Is

This note tracks the stronger rerun path under the updated `configs/model.pilot.yaml` protocol:

- `epochs=4`
- balanced class weighting
- source-balanced sampling for multitask
- early stopping
- repeated train seeds

It exists to prevent ambiguity between:

1. the **completed legacy pilot evidence** in `reports/multiseed/pilot1k/*`
2. the **in-progress strengthened rerun** in `models/multiseed/pilot1k_v2/*`

## Current Completion Status

Completed:

- `baseline`: `seed-11`, `seed-21`, `seed-42`
- `single-task-sentiment`: `seed-11`, `seed-21`, `seed-42`
- `single-task-authenticity`: `seed-11`, `seed-21`, `seed-42`

Not yet completed:

- `multitask`: only `seed-11` is currently present
- no aggregated `summary.json`, `statistics_summary.json`, or final ablation package yet for `pilot1k_v2`

Because of this, `pilot1k_v2` must **not** yet be cited as a completed comparative package in the manuscript.

## Early Observations

The completed single-task runs already show that the stronger protocol materially changes the old pilot picture.

### Single-task Sentiment Macro-F1

- `seed-11`: `0.8789`
- `seed-21`: `0.8723`
- `seed-42`: `0.8385`

### Single-task Authenticity Macro-F1

- `seed-11`: `0.9108`
- `seed-21`: `0.9103`
- `seed-42`: `0.9002`

### Multitask Seed-11 Snapshot

- sentiment Macro-F1: `0.8276`
- authenticity Macro-F1: `0.8694`

These numbers are promising as engineering evidence that the stronger protocol is active and materially different from the legacy one. They are **not yet sufficient** for a new article-level claim, because the multitask side has not completed the full multi-seed sweep.

## Current Interpretation

At this stage, the strongest defensible reading is:

- the repository now contains a **completed legacy pilot package** and an **in-progress stronger rerun path**;
- the stronger rerun already shows that model behavior changes substantially under the updated protocol;
- until all multitask seeds finish and the new aggregate reports are built, the strengthened rerun should be treated as an intermediate validation layer, not as the new central evidence package.

## What Must Happen Next

1. finish `multitask` for all target seeds;
2. generate new multi-seed summary, statistical comparison, and ablation artifacts for `pilot1k_v2`;
3. compare the strengthened rerun against the legacy pilot package;
4. only then decide whether the manuscript's main empirical layer should be replaced or narrowed further.
