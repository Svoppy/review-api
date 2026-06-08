# Runbook финальных экспериментов

Этот файл нужен для того, чтобы довести репозиторий до состояния, когда результаты для статьи собираются последовательно и без импровизации.

## 1. Подготовка данных

### Нормализация корпусов

```bash
PYTHONPATH=src python -m reviewguard.data normalize --dataset rureviews --input data/raw/rureviews/rureviews.csv --output data/processed/rureviews.jsonl --format jsonl
PYTHONPATH=src python -m reviewguard.data normalize --dataset perekrestok --input data/raw/perekrestok/reviews.jsonl --output data/processed/perekrestok.jsonl --format jsonl
PYTHONPATH=src python -m reviewguard.data normalize --dataset opspam --input data/raw/opspam --output data/processed/opspam.jsonl --format jsonl
PYTHONPATH=src python -m reviewguard.data normalize --dataset maide_up --input data/raw/maide_up/maide_up.jsonl --output data/processed/maide_up.jsonl --format jsonl
```

### Сборка joint corpus

```bash
PYTHONPATH=src python -m reviewguard.data merge --inputs data/processed/rureviews.jsonl data/processed/perekrestok.jsonl data/processed/opspam.jsonl data/processed/maide_up.jsonl --output data/processed/joint_reviews.full.jsonl --format jsonl
```

## 2. Pilot protocol

### Baseline

```bash
PYTHONPATH=src python -m reviewguard.training baseline --input data/processed/joint_reviews.pilot1k.jsonl --export-dir models/pilot-baseline
```

### Single-task sentiment

```bash
PYTHONPATH=src python -m reviewguard.training single-task --task sentiment --input data/processed/joint_reviews.pilot1k.jsonl --export-dir models/pilot-single-task-sentiment --config configs/model.pilot.yaml
```

### Single-task authenticity

```bash
PYTHONPATH=src python -m reviewguard.training single-task --task authenticity --input data/processed/joint_reviews.pilot1k.jsonl --export-dir models/pilot-single-task-authenticity --config configs/model.pilot.yaml
```

### Multitask

```bash
PYTHONPATH=src python -m reviewguard.training multitask --input data/processed/joint_reviews.pilot1k.jsonl --export-dir models/pilot-multitask --config configs/model.pilot.yaml
```

## 3. Full protocol

### Baseline

```bash
PYTHONPATH=src python -m reviewguard.training baseline --input data/processed/joint_reviews.full.jsonl --export-dir models/final-baseline
```

### Single-task

```bash
PYTHONPATH=src python -m reviewguard.training single-task --task sentiment --input data/processed/joint_reviews.full.jsonl --export-dir models/final-single-task-sentiment --config configs/model.multitask.yaml
PYTHONPATH=src python -m reviewguard.training single-task --task authenticity --input data/processed/joint_reviews.full.jsonl --export-dir models/final-single-task-authenticity --config configs/model.multitask.yaml
```

### Multitask

```bash
PYTHONPATH=src python -m reviewguard.training multitask --input data/processed/joint_reviews.full.jsonl --export-dir models/final-multitask --config configs/model.multitask.yaml
```

## 4. Что собирать после каждого прогона

- `manifest.json`
- `metadata.json`
- `train_report.json`
- итоговые метрики validation и test
- сведения о `model_name`, `max_length`, `batch_size`, `learning_rate`, `epochs`, `random_state`

## 5. Что обязательно вставить в статью

- итоговую таблицу сравнения моделей;
- отдельную таблицу pilot results и full results;
- минимум одну confusion matrix для sentiment;
- минимум одну confusion matrix для authenticity;
- краткий error analysis;
- ограничения и threats to validity;
- описание точного training protocol.

## 6. Минимальный критерий завершения

Работа по экспериментам считается завершённой, когда выполнены все условия:

1. есть baseline, single-task и multitask reports;
2. все итоговые таблицы заполнены в `docs/results_tables_template_ru.md`;
3. статья `docs/article_final_ru.md` обновлена фактическими результатами;
4. выводы статьи ссылаются на реальные числа, а не на ожидаемое поведение архитектуры.
