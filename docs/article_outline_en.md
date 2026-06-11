# Article Outline

## Working Titles

1. `Toward a Pilot Study of a Multitask Transformer and Web System for Joint Sentiment Analysis and Review Authenticity Detection in E-Commerce`
2. `A Multitask Transformer and Web System for Joint Sentiment Analysis and Review Authenticity Detection in E-Commerce: A Controlled Low-Resource Study`
3. `Conditional Multitask Transfer for Sentiment and Review Authenticity Detection under a Controlled Low-Resource Setting`

## Recommended Paper Structure

1. Title
2. Authors and affiliations
3. Abstract
4. Keywords
5. Introduction
6. Related Work
7. Problem Definition and Research Hypotheses
8. Data and Experimental Materials
9. Methodology
10. Web System Architecture
11. Experimental Design
12. Results and Interpretation
13. Reproducibility / Audit Appendix
14. Threats to Validity and Limitations
15. Conclusion
16. References

## Central Research Idea

Instead of training two fully independent models, the work proposes a shared Transformer encoder with two task-specific heads:

- `sentiment classification`
- `review authenticity detection`

This design aims to:

- exploit shared linguistic signals across both tasks;
- reduce inference cost relative to two standalone models;
- test whether multitask learning yields positive or asymmetric transfer;
- support a single deployable review-analysis service for e-commerce.

## Positioning Statement

The article should be positioned not as “another review classifier” and not as a benchmark paper, but as a bounded empirical study at the intersection of:

1. `multitask learning` for related NLP tasks;
2. `transformer-based review modeling` for e-commerce text;
3. `deceptive/fake review detection`;
4. `AI-generated review detection` as an emerging threat model.

Recommended one-sentence positioning:

> This work addresses the gap between review sentiment modeling and authenticity detection in e-commerce by presenting a multitask Transformer, a partially labeled heterogeneous dataset format, and a reproducible deployment-oriented pipeline evaluated under a controlled low-resource protocol.

## Main Contribution Claims

1. A joint formulation of sentiment analysis and authenticity detection for review-level e-commerce NLP.
2. A unified partially labeled data schema for heterogeneous public datasets.
3. A comparison framework across `classical baseline`, `single-task Transformer`, and `multitask Transformer`.
4. An empirical finding of asymmetric transfer under the executed low-resource mixed setup: authenticity benefits, while sentiment does not.
5. An end-to-end applied system from data preprocessing to API and web UI.
6. A lightweight transparency layer as a deployment-facing usability feature, not as a scientific contribution.

## Research Gaps the Article Can Claim

1. Direct `joint sentiment + authenticity` modeling for e-commerce reviews remains underexplored.
2. Authenticity is often reduced to a simple `fake/real` binary label, while real-world review manipulation is more diverse.
3. Cross-domain and cross-generator robustness remains a major weakness in recent review authenticity research.
4. Many published systems focus either on model quality or deployment engineering, but not both together.
5. Sentiment is rarely treated as an auxiliary task that may improve authenticity detection.
6. Few studies explicitly distinguish what is demonstrated in a bounded study from what still requires benchmark-scale validation.

## Main Real Datasets

- `RuReviews`
- `Perekrestok Reviews`
- `FraudDataset (Yelp)`
- `OpSpam`
- `MAiDE-up`
- optional auxiliary corpus: `Amazon Reviews 2023`

## Current Verified Scope

- local preparation confirmed in this workspace: `RuReviews`, `Perekrestok Reviews`, `MAiDE-up`
- current reported study evidence: sampled subsets of `RuReviews` and `MAiDE-up`
- planned but not yet part of the reported package: `OpSpam`, `FraudDataset (Yelp)`

## Evaluation Metrics

- `Accuracy`
- `Macro-F1`
- `Weighted-F1`
- `Precision`
- `Recall`
- `Confusion Matrix`
- slice-based robustness:
  `mean source-wise Macro-F1`
  `worst-slice Macro-F1`
  `best-minus-worst robustness gap`

## Suggested Tables

1. Dataset comparison table
2. Baseline vs single-task vs multitask results
3. Cross-domain robustness comparison
4. Ablation study on loss weights / task balancing
5. Error analysis by confusion type
6. Split-level class distribution and majority baseline
7. Audit trail for the final experiment bundle

## Suggested Figures

1. End-to-end system architecture
2. Multitask model architecture
3. Training and deployment pipeline
4. Confusion matrices for both tasks

## Important Writing Rules

- Do not invent final metrics before the full experiments are completed.
- Explicitly distinguish gold-label, silver-label, and AI-generated review datasets.
- Do not overclaim that public authenticity datasets represent perfect ground truth.
- State clearly when a result is expected, planned, or hypothetical rather than already measured.
- Treat the current empirical package as a `bounded low-resource study`, not as a benchmark-wide claim.
- Do not let the title, abstract, or conclusion imply general multitask superiority.
- If `H3` is discussed, define robustness operationally rather than rhetorically.
