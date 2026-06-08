# Final Results Table Templates

Use this file as a staging area for the final dissertation metrics before copying the validated numbers into [article_final_en.tex](C:/projects/aitu/review-api/docs/article_final_en.tex).

## 1. Split-Level Class Distribution

| Task | Class | Train | Validation | Test |
|---|---|---:|---:|---:|
| Sentiment | `negative` |  |  |  |
| Sentiment | `neutral` |  |  |  |
| Sentiment | `positive` |  |  |  |
| Authenticity | `authentic` |  |  |  |
| Authenticity | `fake` |  |  |  |

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

## 6. Loss-Weight Ablation

| `lambda_sentiment` | `lambda_authenticity` | Sentiment Macro-F1 | Authenticity Macro-F1 | Comment |
|---:|---:|---:|---:|---|
| `1.0` | `1.0` |  |  | balanced default |
| `0.5` | `1.0` |  |  | authenticity-prioritized |
| `1.0` | `0.5` |  |  | sentiment-prioritized |
| `2.0` | `1.0` |  |  | stronger sentiment weight |
| `1.0` | `2.0` |  |  | stronger authenticity weight |

## 7. Parameter-Sharing Ablation

| Architecture | Sentiment Macro-F1 | Authenticity Macro-F1 | Comment |
|---|---:|---:|---|
| `Separate encoders` |  |  | non-multitask reference |
| `Hard sharing` |  |  | current target design |
| `Soft sharing / task projections` |  |  | optional if implemented |

## 8. Error Analysis

| Example | Task | Gold label | Predicted label | Why the model likely failed |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |

## 9. Audit Trail

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
