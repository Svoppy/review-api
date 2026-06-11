# Submission Checklist

Updated: `2026-06-11`

## 1. Already Ready

- a working end-to-end pipeline `data -> training -> export -> API -> web UI`;
- a complete pilot package: `baseline`, `single-task`, `multitask`, `3-seed` summary, statistics, task ablation;
- leakage-aware split logic and article-grade dataset audit;
- locally built corpora `joint_reviews.current.jsonl` and `joint_reviews.balanced6k.jsonl`;
- the main English manuscripts: [article_draft_en.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_draft_en.md) and [article_final_en.tex](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_final_en.tex).

## 2. Must-Fix Items Before Submission

### Experiments

- regenerate the key baseline / single-task / multitask runs under the current protocol from `configs/model.pilot.yaml`;
- either keep the article explicitly scoped as a bounded pilot or add at least one non-`MAiDE-up` authenticity benchmark to the executed evidence;
- build an article-facing robustness package for `source`, `domain`, and `language`;
- either complete the `source-balanced vs naive` and `loss-weight` ablations or remove them from the main narrative.

### Results Analysis

- include confusion matrices in the article or appendix;
- include a short error analysis;
- keep the canonical appendix [article_results_package_en.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_results_package_en.md) current;
- state class support and the `neutral` / authenticity-coverage limitations explicitly.

### Reproducibility

- keep `docs/reproducibility_note_en.md`, `docs/tested_environment_en.md`, the runbook, and the README aligned with the actual snapshot;
- fix one environment of record for article results;
- remove or rewrite commands that depend on absent raw paths unless they are explicitly marked as optional or local-only.

### Manuscript Text

- keep the backbone narrative aligned with the actually reported pilot results;
- replace plan-like wording with completed-study wording wherever results already exist;
- avoid hidden broad claims about robustness or universal multitask superiority.

## 3. What Not To Do

- do not present the current pilot as a completed multi-benchmark study;
- do not present the `MAiDE-up`-centric authenticity result as a general e-commerce authenticity benchmark;
- do not claim that the current package equally validates human deception, silver fraud, and AI-generated reviews;
- do not describe the transparency layer as a complete explainability system;
- do not invent final numbers without new runs under the current protocol.

## 4. Minimum Strong Submission Bundle

- final synchronized manuscript;
- canonical appendix generated from real report artifacts;
- model comparison table `baseline / single-task / multitask`;
- robustness table by `source/domain/language`;
- confusion matrices and short error analysis;
- limitations / threats to validity section;
- reproducible commands for the main experimental path.

## 5. Readiness Criterion

The work is close to submission-ready when all of the following are true:

1. central experimental claims are backed by real artifacts in `reports/` and `models/`;
2. the manuscript does not promise more than the current benchmark bundle supports;
3. reproducibility docs match the actual environment and commands;
4. the pilot package is either honestly positioned as a bounded study or expanded into a broader authenticity benchmark;
5. every major result claim is supported by a table, metric, or figure in the manuscript or canonical appendix.
