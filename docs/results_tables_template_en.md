# Final Results Table Templates

Use this file as a staging area for the validated English-language metrics before copying them into [article_final_en.tex](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_final_en.tex).

## Reviewer note

This template is intentionally stricter than a simple score sheet. The goal is to prevent weak reporting such as:

- pooled authenticity numbers without source-specific breakdowns;
- Macro-F1 claims built on single-digit minority-class support;
- comparison tables without majority baselines, seeds, or audit metadata.

## 1. Split-Level Class Distribution

| Task | Class | Train | Validation | Test |
|---|---|---:|---:|---:|
| Sentiment | `negative` |  |  |  |
| Sentiment | `neutral` |  |  |  |
| Sentiment | `positive` |  |  |  |
| Authenticity | `authentic` |  |  |  |
| Authenticity | `fake` |  |  |  |

Add one line of interpretation below the filled table:

- whether any class has critically low support;
- whether the split is still suitable for reviewer-facing Macro-F1 claims.

## 2. Majority Baseline

| Task | Majority class | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Sentiment |  |  |  |
| Authenticity |  |  |  |

## 3. Main Comparative Results

| Model | Task | Accuracy | Macro-F1 | Weighted-F1 | Precision_macro | Recall_macro | Mean ± Std | Significance note |
|---|---|---:|---:|---:|---:|---:|---|---|
| `TF-IDF + Logistic Regression` | Sentiment |  |  |  |  |  |  |  |
| `Single-task Transformer` | Sentiment |  |  |  |  |  |  |  |
| `Multitask Transformer` | Sentiment |  |  |  |  |  |  |  |
| `TF-IDF + Logistic Regression` | Authenticity |  |  |  |  |  |  |  |
| `Single-task Transformer` | Authenticity |  |  |  |  |  |  |  |
| `Multitask Transformer` | Authenticity |  |  |  |  |  |  |  |

Minimum reporting rule:

- for Transformer rows, use `mean ± std` across at least 3 train seeds;
- if a claim such as “better” or “more robust” appears in the article, the significance note should mention the corresponding CI and p-value or explicitly say `no clear difference`.

## 4. Pilot-Protocol Comparison

| Model | Sentiment Accuracy | Sentiment Macro-F1 | Authenticity Accuracy | Authenticity Macro-F1 | Comment |
|---|---:|---:|---:|---:|---|
| `TF-IDF + Logistic Regression` | `0.8000` | `0.7368` | `0.8000` | `0.7999` | already obtained |
| `Single-task Transformer` |  |  |  |  |  |
| `Multitask Transformer` |  |  |  |  |  |

## 5. Dataset-Specific Authenticity Results

| Source | Model | Accuracy | Macro-F1 | Comment |
|---|---|---:|---:|---|
| `MAiDE-up` | `TF-IDF + Logistic Regression` |  |  | AI-generated reviews |
| `MAiDE-up` | `Single-task Transformer` |  |  | AI-generated reviews |
| `MAiDE-up` | `Multitask Transformer` |  |  | AI-generated reviews |
| `OpSpam` | `Single-task Transformer` |  |  | human deceptive reviews |
| `OpSpam` | `Multitask Transformer` |  |  | human deceptive reviews |
| `FraudDataset (Yelp)` | `Single-task Transformer` |  |  | silver fraud labels |
| `FraudDataset (Yelp)` | `Multitask Transformer` |  |  | silver fraud labels |

Interpretation rule:

- do not collapse these sources into one headline authenticity claim without preserving this breakdown in the article or appendix.

## 6. Slice-Based Robustness

| Task | Slice field | Mean slice Macro-F1 | Worst-slice Macro-F1 | Best-slice Macro-F1 | Robustness gap | Comment |
|---|---|---:|---:|---:|---:|---|
| Sentiment | `source` |  |  |  |  |  |
| Sentiment | `domain` |  |  |  |  |  |
| Sentiment | `language` |  |  |  |  |  |
| Authenticity | `source` |  |  |  |  |  |
| Authenticity | `domain` |  |  |  |  |  |
| Authenticity | `language` |  |  |  |  |  |

Use this table to operationalize `H3`. A claim of improved robustness should be based on this table, not on pooled accuracy alone.

## 7. Loss-Weight Ablation

| `lambda_sentiment` | `lambda_authenticity` | Sentiment Macro-F1 | Authenticity Macro-F1 | Comment |
|---:|---:|---:|---:|---|
| `1.0` | `1.0` |  |  | balanced default |
| `0.5` | `1.0` |  |  | authenticity-prioritized |
| `1.0` | `0.5` |  |  | sentiment-prioritized |
| `2.0` | `1.0` |  |  | stronger sentiment weight |
| `1.0` | `2.0` |  |  | stronger authenticity weight |

## 8. Parameter-Sharing Ablation

| Architecture | Sentiment Macro-F1 | Authenticity Macro-F1 | Comment |
|---|---:|---:|---|
| `Separate encoders` |  |  | non-multitask reference |
| `Hard sharing` |  |  | current target design |
| `Soft sharing / task projections` |  |  | optional if implemented |

## 9. Error Analysis

| Example | Task | Gold label | Predicted label | Why the model likely failed |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |

## 10. Audit Trail

| Field | Value |
|---|---|
| Prepared input bundle |  |
| Input artifact hash |  |
| Split policy |  |
| Leakage-check outcome |  |
| Random seeds |  |
| Backbone |  |
| Optimization setup |  |
| Checkpoint manifest |  |
| Training command |  |
| Evaluation command |  |
| Duplicate / leakage audit summary |  |

Before promoting numbers into the article, confirm:

- the exact dataset bundle is stated;
- the split and seed protocol is stated;
- the minority-class support is not silently weak;
- any robustness claim is backed by the slice table above.
