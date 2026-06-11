# Reproducibility Note

Verification date: `2026-06-11`

## 1. What Is Already Reproducible

- package import and module structure for `reviewguard`;
- the data pipeline: normalization, merge, split, and audit;
- metric computation, the statistical layer, and robustness utilities;
- baseline / single-task / multitask training CLIs at the command-and-config level;
- exported artifact format and API wiring;
- the current automated test suite.

## 2. What Was Actually Verified in the Current Snapshot

Command:

```bash
PYTHONPATH=src .venv314/bin/python -m pytest -q
```

Result:

- `62 passed in 3.64s`

This means the current snapshot is reproducible at the level of local package logic, tests, and the main research utility layers.

## 3. What Still Limits Full Article-Level Reproducibility

- the reported pilot artifacts were not originally generated under the same strengthened protocol now described as the minimum standard in `configs/model.pilot.yaml` and `docs/final_experiment_runbook_ru.md`;
- the local workspace still lacks prepared raw/export corpora for `OpSpam` and `FraudYelpDataset`, so the completed empirical authenticity scope remains narrower than the intended final benchmark;
- some raw-data commands in the documentation depend on locally prepared files and should not be interpreted as guaranteed zero-setup commands for every snapshot;
- the environment of record for already saved model artifacts and the current pinned environment still need additional alignment before a stronger submission-ready reproducibility claim.

## 4. Correct Wording for the Manuscript

At the current stage, the defensible claim is:

> The repository contains a reproducible research pipeline and a fully passing local test package for the codebase. At the same time, full reproducibility of the article-level empirical results still requires environment-of-record alignment, rerunning the key model comparisons under the current strengthened protocol, and expanding the locally prepared benchmark bundle.

## 5. What Is Still Needed for a Stronger Submission Claim

1. Regenerate the key baseline / single-task / multitask runs under the current protocol with `epochs=4`, `balanced`, `source-balanced`, and `early stopping`.
2. Fix one canonical environment of record for article results.
3. Synchronize runbook commands with the local paths and prepared datasets that actually exist.
4. Extend the local authenticity bundle with at least one non-`MAiDE-up` benchmark.
