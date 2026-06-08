# Roadmap

## Phase 1

- finalize dataset choices
- implement shared data schema
- add raw-to-unified preprocessing loaders for `RuReviews`, `Perekrestok`, `OpSpam`, and `MAiDE-up`-shaped files
- write normalized `jsonl` and `csv` outputs for multitask training
- status: implemented in the repository

## Phase 2

- train classical baselines
- train single-task Transformer baselines
- define experiment tracking tables
- status: classical baseline scaffold implemented, single-task Transformer experiments still pending

## Phase 3

- train multitask Transformer
- tune loss weights
- compare with baselines
- status: multitask training scaffold implemented, real experiments still pending

## Phase 4

- connect trained checkpoint to API
- expose probabilities and explanations
- test end-to-end through the web UI
- status: checkpoint loading, inference wiring, and lightweight probability/explanation layer implemented; end-to-end validation with a trained checkpoint is still pending

## Phase 5

- prepare dissertation figures
- write methodology and results chapters
- perform ablation, error analysis, and reproducibility packaging for the defense build
