# Пакет итоговых таблиц статьи

Этот файл собирается автоматически из текущих отчетов и служит canonical article-facing appendix для текущего snapshot.

## Источники

- pilot summary: `reports/multiseed/pilot1k/summary.json`
- statistics summary: `reports/multiseed/pilot1k_statistics/statistics_summary.json`
- balanced audit: `reports/audit/joint_reviews.balanced6k.audit.json`

## Таблица 1. Снимок датасетов и текущего локального статуса

| Датасет | Язык | Домен | Размер локальной подготовки | Тип меток | Задача | Ограничения |
|---|---|---|---:|---|---|---|
| `RuReviews` | RU | e-commerce | 60602 | готовые sentiment labels | Sentiment | нет authenticity label coverage |
| `Perekrestok Reviews` | RU | retail | 642682 | rating-derived sentiment | Sentiment | authenticity отсутствует; sentiment partially heuristic |
| `OpSpam` | EN | hospitality | not prepared locally | truthful/deceptive labels | Authenticity | planned extension; raw text not prepared locally |
| `FraudDataset (Yelp)` | EN | local commerce | not prepared locally | silver fraud labels | Authenticity | adapter ready, local text export not prepared |
| `MAiDE-up` | multilingual | hospitality | 19985 | sentiment + AI-generated authenticity | Sentiment + Authenticity | current authenticity evidence comes only from this family |

## Таблица 2. Основное сравнение моделей на выполненном pilot protocol

| Модель | Корпус | Задача | Accuracy | Macro-F1 | Weighted-F1 | Precision_macro | Recall_macro | Mean +/- Std | Значимость |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| TF-IDF + Logistic Regression | pilot1k | Sentiment | 0.8000 | 0.7368 | 0.8002 | 0.7385 | 0.7370 | 0.7368 +/- 0.0000 | референсный classical baseline |
| Single-task Transformer | pilot1k | Sentiment | 0.8150 | 0.5814 | 0.8063 | 0.6024 | 0.5782 | 0.5814 +/- 0.0526 | референсный Transformer comparison |
| Multitask Transformer | pilot1k | Sentiment | 0.7317 | 0.5856 | 0.7234 | 0.6163 | 0.5841 | 0.5856 +/- 0.0665 | сопоставимо с single-task, хуже baseline (p=0.9510; p=0.0025) |
| TF-IDF + Logistic Regression | pilot1k | Authenticity | 0.8000 | 0.7999 | 0.7999 | 0.8005 | 0.8000 | 0.7999 +/- 0.0000 | референсный classical baseline |
| Single-task Transformer | pilot1k | Authenticity | 0.8400 | 0.8328 | 0.8328 | 0.8740 | 0.8400 | 0.8328 +/- 0.1075 | референсный Transformer comparison |
| Multitask Transformer | pilot1k | Authenticity | 0.9167 | 0.9160 | 0.9160 | 0.9248 | 0.9167 | 0.9160 +/- 0.0524 | лучше single-task и baseline (p=0.0005; p=0.0005) |

## Таблица 3. Парные сравнения по Macro-F1 на фиксированном test split

| Задача | Модель A | Модель B | Delta | 95% CI | p-value |
|---|---|---|---:|---|---:|
| Authenticity | multitask | single-task-authenticity | 0.0832 | [0.0447, 0.1274] | 0.0005 |
| Authenticity | multitask | baseline | 0.1160 | [0.0447, 0.1988] | 0.0005 |
| Sentiment | multitask | single-task-sentiment | 0.0042 | [-0.0915, 0.0954] | 0.9510 |
| Sentiment | multitask | baseline | -0.1513 | [-0.2504, -0.0446] | 0.0025 |

## Таблица 4. Устойчивость по train seeds

| Модель | Задача | Mean Accuracy | Std Accuracy | Mean Macro-F1 | Std Macro-F1 |
|---|---|---:|---:|---:|---:|
| Single-task Transformer | Sentiment | 0.8150 | 0.0132 | 0.5814 | 0.0526 |
| Multitask Transformer | Sentiment | 0.7317 | 0.0355 | 0.5856 | 0.0665 |
| Single-task Transformer | Authenticity | 0.8400 | 0.0954 | 0.8328 | 0.1075 |
| Multitask Transformer | Authenticity | 0.9167 | 0.0513 | 0.9160 | 0.0524 |

## Таблица 5. Расширенный balanced6k snapshot: data-readiness и leakage audit

- records: `6000`
- source distribution: `rureviews=2500`, `perekrestok=2500`, `maide_up=1000`
- split overlap: `train/valid normalized=0`, `train/test normalized=0`, `valid/test normalized=0`
- duplicate summary: `exact=81`, `normalized=97`
- authenticity source coverage: `maide_up=1.0`, `rureviews=0.0`, `perekrestok=0.0`

## Таблица 6. Классовый баланс и majority baseline на balanced6k test split

| Корпус | Задача | Класс | Число объектов | Доля |
|---|---|---|---:|---:|
| `balanced6k` | Sentiment | `negative` | 187 | 0.3086 |
| `balanced6k` | Sentiment | `neutral` | 8 | 0.0132 |
| `balanced6k` | Sentiment | `positive` | 411 | 0.6782 |
| `balanced6k` | Authenticity | `authentic` | 51 | 0.5100 |
| `balanced6k` | Authenticity | `fake` | 49 | 0.4900 |

| Корпус | Задача | Majority baseline Accuracy | Majority baseline Macro-F1 |
|---|---|---:|---:|
| `balanced6k` | Sentiment | 0.6782 | 0.2694 |
| `balanced6k` | Authenticity | 0.5100 | 0.3377 |

## Таблица 7. Representative confusion matrices на pilot checkpoints

### Baseline sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 77 | 0 | 22 |
| `neutral` | 0 | 3 | 2 |
| `positive` | 14 | 2 | 80 |

### Single-task sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 89 | 0 | 10 |
| `neutral` | 4 | 0 | 1 |
| `positive` | 24 | 0 | 72 |

### Multitask sentiment

| gold ↓ / pred → | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 86 | 3 | 10 |
| `neutral` | 2 | 2 | 1 |
| `positive` | 43 | 1 | 52 |

### Multitask authenticity

| gold ↓ / pred → | `authentic` | `fake` |
|---|---|---|
| `authentic` | 37 | 13 |
| `fake` | 1 | 49 |

### Single-task authenticity

| gold ↓ / pred → | `authentic` | `fake` |
|---|---|---|
| `authentic` | 23 | 27 |
| `fake` | 0 | 50 |

## Краткий error analysis

- baseline по sentiment корректно восстанавливает `neutral` только в `3/5` pilot-примерах;
- single-task sentiment фактически рушится на редком классе `neutral`: только `0/5` верных ответов;
- multitask sentiment все еще переоценивает `negative` для положительных отзывов (`positive -> negative = 43`);
- multitask authenticity пропускает только `1` fake-review на representative pilot-checkpoint;
- single-task authenticity переоценивает класс `fake`, давая `27` ошибок `authentic -> fake`;

## Что уже article-ready

- pilot comparison `baseline / single-task / multitask` собран на одном фиксированном split;
- для pilot уже есть `mean +/- std`, bootstrap CI и approximate randomization tests;
- balanced6k snapshot уже проходит leakage-safe split audit с нулевым `normalized_text_overlap` между split'ами;
- provenance и label coverage теперь явно фиксируются в unified pipeline.

## Что еще не дает честно ставить 10/10

- authenticity coverage в текущем expanded snapshot по-прежнему идет только из `MAiDE-up`;
- в balanced6k test split по sentiment все еще только `8` примеров класса `neutral`;
- полные article-grade robustness и ablation reruns по усиленному протоколу еще не выполнены.
