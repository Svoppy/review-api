# Шаблоны итоговых таблиц результатов

Использовать этот файл как заготовку для финального заполнения после завершения прогонов.

## Таблица 1. Сравнение датасетов

| Датасет | Язык | Домен | Размер | Тип меток | Задача | Ограничения |
|---|---|---|---:|---|---|---|
| `RuReviews` |  |  |  |  |  |  |
| `Perekrestok Reviews` |  |  |  |  |  |  |
| `OpSpam` |  |  |  |  |  |  |
| `FraudDataset (Yelp)` |  |  |  |  |  |  |
| `MAiDE-up` |  |  |  |  |  |  |

## Таблица 2. Основное сравнение моделей

| Модель | Корпус | Задача | Accuracy | Macro-F1 | Weighted-F1 | Precision_macro | Recall_macro | Mean ± Std | Значимость |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `TF-IDF + Logistic Regression` |  | Sentiment |  |  |  |  |  |  |  |
| `Single-task Transformer` |  | Sentiment |  |  |  |  |  |  |  |
| `Multitask Transformer` |  | Sentiment |  |  |  |  |  |  |  |
| `TF-IDF + Logistic Regression` |  | Authenticity |  |  |  |  |  |  |  |
| `Single-task Transformer` |  | Authenticity |  |  |  |  |  |  |  |
| `Multitask Transformer` |  | Authenticity |  |  |  |  |  |  |  |

## Таблица 3. Сравнение на pilot protocol

| Модель | Sentiment Accuracy | Sentiment Macro-F1 | Authenticity Accuracy | Authenticity Macro-F1 | Примечание |
|---|---:|---:|---:|---:|---|
| `TF-IDF + Logistic Regression` | `0.8000` | `0.7368` | `0.8000` | `0.7999` | уже получено |
| `Single-task Transformer` |  |  |  |  |  |
| `Multitask Transformer` |  |  |  |  |  |

## Таблица 4. Абляция по весам функций потерь

| `λ_sentiment` | `λ_authenticity` | Sentiment Macro-F1 | Authenticity Macro-F1 | Комментарий |
|---:|---:|---:|---:|---|
| `1.0` | `1.0` |  |  | baseline |
| `0.5` | `1.0` |  |  |  |
| `1.0` | `0.5` |  |  |  |
| `2.0` | `1.0` |  |  |  |
| `1.0` | `2.0` |  |  |  |

## Таблица 5. Domain robustness

| Train Sources | Test Source | Задача | Accuracy | Macro-F1 | Наблюдение |
|---|---|---|---:|---:|---|
|  |  | Sentiment |  |  |  |
|  |  | Authenticity |  |  |  |

## Таблица 6. Ошибки модели

| Пример | Истинная метка | Предсказание | Задача | Возможная причина ошибки |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

## Таблица 7. Средние результаты по нескольким seeds

| Модель | Задача | Mean Accuracy | Std Accuracy | Mean Macro-F1 | Std Macro-F1 |
|---|---|---:|---:|---:|---:|
| `Single-task Transformer` | Sentiment |  |  |  |  |
| `Multitask Transformer` | Sentiment |  |  |  |  |
| `Single-task Transformer` | Authenticity |  |  |  |  |
| `Multitask Transformer` | Authenticity |  |  |  |  |

## Таблица 8. Распределение классов и majority baseline

| Корпус | Задача | Класс | Число объектов | Доля |
|---|---|---|---:|---:|
|  | Sentiment | `negative` |  |  |
|  | Sentiment | `neutral` |  |  |
|  | Sentiment | `positive` |  |  |
|  | Authenticity | `authentic` |  |  |
|  | Authenticity | `fake` |  |  |

| Корпус | Задача | Majority baseline Accuracy | Majority baseline Macro-F1 |
|---|---|---:|---:|
|  | Sentiment |  |  |
|  | Authenticity |  |  |
