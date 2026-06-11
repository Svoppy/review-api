# Runbook финальных экспериментов

Этот файл нужен для того, чтобы довести репозиторий до состояния, когда результаты для статьи собираются последовательно и без импровизации.

## 1. Подготовка данных

Этот runbook разделяет два режима:

- `snapshot-ready path`: команды, которые уже воспроизводимы на текущем workspace из подготовленных `data/processed/*`;
- `full raw-data path`: команды, требующие локально подготовленных raw/export файлов и не гарантирующие запуск “с нуля” на любом snapshot.

### Snapshot-ready path

Если задача состоит в воспроизводимом локальном rerun на текущем snapshot, опирайтесь на уже подготовленные processed corpora:

```bash
PYTHONPATH=src .venv314/bin/python -m reviewguard.data merge --inputs data/processed/rureviews.jsonl data/processed/perekrestok.jsonl data/processed/maide_up.jsonl --output data/processed/joint_reviews.current.jsonl --format jsonl
PYTHONPATH=src .venv314/bin/python scripts/sample_unified_dataset.py \
  --input data/processed/joint_reviews.current.jsonl \
  --output data/processed/joint_reviews.balanced6k.jsonl \
  --max-rows 6000 \
  --seed 42 \
  --stratify-fields source sentiment_label authenticity_label language domain \
  --min-per-group 40 \
  --source-cap rureviews=2500 \
  --source-cap perekrestok=2500 \
  --source-cap maide_up=1000
```

### Full raw-data path

Следующие команды относятся к сценарию, где raw/export files уже подготовлены локально в ожидаемых путях:

```bash
PYTHONPATH=src .venv314/bin/python -m reviewguard.data normalize --dataset rureviews --input data/raw/rureviews/rureviews.csv --output data/processed/rureviews.jsonl --format jsonl
PYTHONPATH=src .venv314/bin/python -m reviewguard.data normalize --dataset perekrestok --input data/raw/perekrestok/reviews.jsonl --output data/processed/perekrestok.jsonl --format jsonl
PYTHONPATH=src .venv314/bin/python -m reviewguard.data normalize --dataset maide_up --input data/raw/maide_up/maide_up.jsonl --output data/processed/maide_up.jsonl --format jsonl
```

### Сборка joint corpus для текущего snapshot

```bash
PYTHONPATH=src python -m reviewguard.data merge --inputs data/processed/rureviews.jsonl data/processed/perekrestok.jsonl data/processed/maide_up.jsonl --output data/processed/joint_reviews.current.jsonl --format jsonl
```

### Сборка увеличенного benchmark-поднабора с контролем редких классов

Вместо старого `1k + 1k` sample для следующего цикла следует собирать более крупный pilot / pre-final benchmark с composite stratification:

```bash
PYTHONPATH=src python scripts/sample_unified_dataset.py \
  --input data/processed/joint_reviews.current.jsonl \
  --output data/processed/joint_reviews.balanced6k.jsonl \
  --max-rows 6000 \
  --seed 42 \
  --stratify-fields source sentiment_label authenticity_label language domain \
  --min-per-group 40 \
  --source-cap rureviews=2500 \
  --source-cap perekrestok=2500 \
  --source-cap maide_up=1000
```

Это не заменяет полный корпус, но убирает главный defect старого tiny pilot:

- в тесте будет больше minority-support;
- `neutral` перестанет определяться пятью примерами;
- агрегированные метрики будут меньше зависеть от случайной выборки;
- mixed-domain корпус станет контролируемым по source composition.

### Audit до любого обучения

До старта модели обязателен split-level audit. В обновлённом протоколе он должен проверять не только размеры сплитов и majority baseline, но и global normalized-text overlap и coverage меток по `source/domain/language`:

```bash
PYTHONPATH=src python -m reviewguard.data audit \
  --input data/processed/joint_reviews.balanced6k.jsonl \
  --output reports/audit/joint_reviews.balanced6k.audit.json \
  --train-size 0.8 \
  --valid-size 0.1 \
  --test-size 0.1 \
  --random-state 42
```

Если audit предупреждает про `pilot-scale`, `macro-F1 is likely unstable`, `normalized text overlap` или mixed-language/domain leakage-risk, такой run не должен считаться финальным article run.

### Условное расширение полного benchmark

Следующие шаги не должны входить в основной runbook, пока соответствующие корпуса не подготовлены локально:

```bash
PYTHONPATH=src python -m reviewguard.data normalize --dataset opspam --input data/raw/opspam --output data/processed/opspam.jsonl --format jsonl
PYTHONPATH=src python -m reviewguard.data normalize --dataset fraudyelp --input data/raw/fraudyelp --output data/processed/fraudyelp.jsonl --format jsonl
PYTHONPATH=src python -m reviewguard.data merge --inputs data/processed/rureviews.jsonl data/processed/perekrestok.jsonl data/processed/maide_up.jsonl data/processed/opspam.jsonl data/processed/fraudyelp.jsonl --output data/processed/joint_reviews.full_authenticity.jsonl --format jsonl
```

Для `FraudDataset (Yelp)` текущий adapter поддерживает только локальный review-text export. Он не должен описываться как прямой loader произвольного raw graph dump без промежуточной подготовки текста.

## 2. Pilot protocol

### Multi-seed sweep на фиксированном split

Для честной `multiple-seed` оценки держим `split seed` фиксированным и варьируем только `train seed`.

```bash
.venv314/bin/python scripts/run_multiseed_experiments.py \
  --input data/processed/joint_reviews.pilot1k.jsonl \
  --config configs/model.pilot.yaml \
  --split-seed 42 \
  --train-seeds 11 21 42 \
  --output-root models/multiseed/pilot1k \
  --report-dir reports/multiseed/pilot1k
```

После завершения должны появиться:

- `models/multiseed/pilot1k/<model>/seed-<n>/train_report.json`
- `reports/multiseed/pilot1k/summary.json`
- `reports/multiseed/pilot1k/metrics_table.csv`
- `reports/multiseed/pilot1k/summary.md`

Начиная со следующего цикла, `1 epoch` больше не используется даже для pilot comparison. Базовый протокол репозитория теперь предполагает:

- `epochs=4`;
- `balanced` class weighting;
- `source_balanced` sampling для multitask;
- `early_stopping_patience=2`;
- multi-seed reporting как минимальный стандарт, а не как optional extension.

### Statistical layer: confidence intervals + paired comparisons

После multi-seed sweep можно собрать статистический пакет для статьи:

```bash
.venv314/bin/python scripts/analyze_multiseed_statistics.py \
  --input data/processed/joint_reviews.pilot1k.jsonl \
  --multiseed-root models/multiseed/pilot1k \
  --summary-json reports/multiseed/pilot1k/summary.json \
  --output-dir reports/multiseed/pilot1k_statistics \
  --split-seed 42 \
  --train-seeds 11 21 42
```

Ожидаемые артефакты:

- `reports/multiseed/pilot1k_statistics/statistics_summary.json`
- `reports/multiseed/pilot1k_statistics/model_intervals.csv`
- `reports/multiseed/pilot1k_statistics/pairwise_comparisons.csv`
- `reports/multiseed/pilot1k_statistics/statistics_report.md`

### Ablation layer: single-task vs multitask

Для article-ready интерпретации архитектурного вопроса отдельно собирается ablation-слой:

```bash
.venv314/bin/python scripts/build_task_ablation_report.py \
  --multiseed-summary reports/multiseed/pilot1k/summary.json \
  --statistics-summary reports/multiseed/pilot1k_statistics/statistics_summary.json \
  --output-dir reports/multiseed/pilot1k_ablation
```

Ожидаемые артефакты:

- `reports/multiseed/pilot1k_ablation/task_ablation.json`
- `reports/multiseed/pilot1k_ablation/task_ablation.csv`
- `reports/multiseed/pilot1k_ablation/task_ablation.md`

### Robustness layer: source / domain / language slices

После каждого train run `train_report.json` теперь уже содержит slice-based robustness summary. Если нужны отдельные артефакты для статьи, можно дополнительно собрать их из экспортированных предсказаний:

- source-wise macro-F1 mean;
- worst-slice macro-F1;
- best-minus-worst robustness gap;
- domain-aware и language-aware breakdown.

Операционализация `H3` должна быть именно такой: robustness = не абстрактная "устойчивость", а наблюдаемое поведение на source/domain/language slices.

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

Этот раздел относится к расширенному benchmark только после фактической локальной интеграции дополнительных authenticity-корпусов. До этого момента нельзя описывать `joint_reviews.full.jsonl` как уже существующий основной экспериментальный ресурс.

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
- сведения о `model_name`, `max_length`, `batch_size`, `learning_rate`, `epochs`, `split_random_state`, `train_random_state`

## 5. Что обязательно вставить в статью

- итоговую таблицу сравнения моделей;
- отдельную таблицу pilot results и full results;
- минимум одну confusion matrix для sentiment;
- минимум одну confusion matrix для authenticity;
- краткий error analysis;
- explicit class-support table для каждого split;
- robustness table по `source`, `domain`, `language`;
- ablation `single-task vs multitask` и `source-balanced vs naive`;
- ограничения и threats to validity;
- описание точного training protocol.

## 6. Минимальный критерий завершения

Работа по экспериментам считается завершённой, когда выполнены все условия:

1. есть baseline, single-task и multitask reports;
2. canonical results appendix собран в `docs/article_results_package_ru.md`;
3. статья `docs/article_final_ru.md` обновлена фактическими результатами;
4. выводы статьи ссылаются на реальные числа, а не на ожидаемое поведение архитектуры.
