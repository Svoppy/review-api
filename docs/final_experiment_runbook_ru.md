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

### Сборка article-grade mixed benchmark с большим test support

Для следующего сильного цикла теперь есть отдельный builder, который формирует более крупный смешанный корпус и принудительно удерживает редкий `neutral` и баланс authenticity:

```bash
PYTHONPATH=src .venv314/bin/python scripts/build_article_benchmark.py \
  --output data/processed/joint_reviews.article20k.jsonl \
  --audit-output reports/audit/joint_reviews.article20k.audit.json
```

На текущем snapshot этот путь уже даёт:

- `701` authenticity-labeled examples в test;
- `122` примера класса `neutral` в sentiment test;
- нулевой `normalized_text_overlap` между split'ами.

### Расширение русскоязычного e-commerce материала Wildberries

Для внешнего sentiment-источника проект поддерживает публичный исследовательский набор Wildberries. Он уже de-identified upstream и имеет лицензию `CC BY-NC-SA 4.0`; используйте его только для research/non-commercial сценария с указанием источника и лицензии.

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets wildberries --output-root data/raw
PYTHONPATH=src python -m reviewguard.data normalize \
  --dataset wildberries \
  --input data/raw/wildberries/wildberries.jsonl \
  --output data/processed/wildberries.jsonl \
  --format jsonl
```

Для сравнения с текущим `article20k` можно собрать отдельный, не заменяющий его `article30k` candidate c 10 тысячами Wildberries отзывов:

```bash
PYTHONPATH=src python scripts/build_article_benchmark.py \
  --output data/processed/joint_reviews.article30k.jsonl \
  --audit-output reports/audit/joint_reviews.article30k.audit.json \
  --source-target perekrestok=10000 \
  --source-target rureviews=3000 \
  --source-target maide_up=7000 \
  --source-target wildberries=10000
```

Wildberries добавляет только rating-derived sentiment. Он не должен получать искусственные `authenticity` labels и не устраняет потребность в независимом fake-review benchmark.

### Apple Silicon: Mac mini M2 Pro с 16 GB unified memory

Для `article30k` используйте отдельный memory-safe профиль `configs/model.article30k.mac_m2_16gb.yaml`: physical batch равен `2`, gradient accumulation равен `4` (эффективный batch `8`), длина текста ограничена `192` токенами, а gradient checkpointing снижает пик памяти. До запуска убедитесь, что `torch.backends.mps.is_available()` возвращает `True`.

### Сборка same-domain benchmark для контроля доменного сдвига

```bash
PYTHONPATH=src .venv314/bin/python scripts/build_article_benchmark.py \
  --output data/processed/joint_reviews.maide7k.jsonl \
  --audit-output reports/audit/joint_reviews.maide7k.audit.json \
  --source-target maide_up=7000 \
  --source-require maide_up:authenticity_label:authentic=3500 \
  --source-require maide_up:authenticity_label:fake=3500
```

Этот корпус нужен не как финальный benchmark, а как контрольный same-domain scenario: обе задачи живут в одном источнике и меньше путают эффект multitask с эффектом domain shift.

### Почему это уже лучше старого tiny pilot

Это ещё не заменяет полный multi-benchmark package, но убирает главный defect старого tiny pilot:

- в тесте будет больше minority-support;
- `neutral` больше не определяется пятью примерами;
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

### Рекомендуемый единый запуск release-пакета

Финальные числа следует получать одним оркестратором, а не последовательностью вручную скопированных команд. Он сохраняет неизменяемую копию YAML-конфига, запускает preflight, `5` seed'ов для всех baseline-моделей, статистический слой и ablation, после чего оставляет `release_manifest.json` со всеми фактическими командами.

Запускать только в среде с CUDA или Apple Metal:

```bash
PYTHONPATH=src python scripts/run_article_release.py
```

На CPU скрипт останавливается до записи model artifacts. Проверить будущий план без GPU и без записи файлов можно так:

```bash
PYTHONPATH=src python scripts/run_article_release.py --dry-run
```

### Multi-seed sweep на фиксированном split

Для честной `multiple-seed` оценки держим `split seed` фиксированным и варьируем только `train seed`. Новый нормальный минимум теперь `5` seeds, а не `3`.

```bash
.venv314/bin/python scripts/run_multiseed_experiments.py \
  --input data/processed/joint_reviews.article20k.jsonl \
  --config configs/model.article.yaml \
  --split-seed 42 \
  --train-seeds 11 21 42 84 126 \
  --output-root models/multiseed/article20k \
  --report-dir reports/multiseed/article20k
```

После завершения должны появиться:

- `models/multiseed/article20k/<model>/seed-<n>/train_report.json`
- `reports/multiseed/article20k/summary.json`
- `reports/multiseed/article20k/metrics_table.csv`
- `reports/multiseed/article20k/summary.md`

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
  --input data/processed/joint_reviews.article20k.jsonl \
  --multiseed-root models/multiseed/article20k \
  --summary-json reports/multiseed/article20k/summary.json \
  --output-dir reports/multiseed/article20k_statistics \
  --split-seed 42 \
  --train-seeds 11 21 42 84 126
```

Ожидаемые артефакты:

- `reports/multiseed/article20k_statistics/statistics_summary.json`
- `reports/multiseed/article20k_statistics/model_intervals.csv`
- `reports/multiseed/article20k_statistics/pairwise_comparisons.csv`
- `reports/multiseed/article20k_statistics/statistics_report.md`

Теперь statistical layer дополнительно пишет `Cohen's dz` и automatic small-sample cautions, чтобы `p-value` не выглядела как ложная строгая истина на маленьком test split.

### Ablation layer: single-task vs multitask

Для article-ready интерпретации архитектурного вопроса отдельно собирается ablation-слой:

```bash
.venv314/bin/python scripts/build_task_ablation_report.py \
  --multiseed-summary reports/multiseed/article20k/summary.json \
  --statistics-summary reports/multiseed/article20k_statistics/statistics_summary.json \
  --output-dir reports/multiseed/article20k_ablation
```

Ожидаемые артефакты:

- `reports/multiseed/article20k_ablation/task_ablation.json`
- `reports/multiseed/article20k_ablation/task_ablation.csv`
- `reports/multiseed/article20k_ablation/task_ablation.md`

### Matrix-runs для `lambda`, backbone и learning curves

Для sweep-пакетов больше не нужно вручную плодить команды. Теперь это делается через один matrix-runner:

```bash
PYTHONPATH=src .venv314/bin/python scripts/run_experiment_matrix.py \
  --input data/processed/joint_reviews.article20k.jsonl \
  --base-config configs/model.article.yaml \
  --output-root models/matrix \
  --report-root reports/matrix \
  --variant label=lambda_03_07,loss_weights.sentiment=0.3,loss_weights.authenticity=0.7 \
  --variant label=lambda_05_05,loss_weights.sentiment=0.5,loss_weights.authenticity=0.5 \
  --variant label=lambda_07_03,loss_weights.sentiment=0.7,loss_weights.authenticity=0.3
```

Для backbone ablation:

```bash
PYTHONPATH=src .venv314/bin/python scripts/run_experiment_matrix.py \
  --input data/processed/joint_reviews.article20k.jsonl \
  --base-config configs/model.article.yaml \
  --output-root models/matrix_backbones \
  --report-root reports/matrix_backbones \
  --variant label=xlmr,model_name=FacebookAI/xlm-roberta-base \
  --variant label=distilbert,model_name=distilbert-base-multilingual-cased
```

Для learning curves сначала готовим corpora:

```bash
PYTHONPATH=src .venv314/bin/python scripts/build_learning_curve_corpora.py \
  --input data/processed/joint_reviews.article20k.jsonl \
  --output-root data/processed/learning_curves
```

После этого соответствующие `.jsonl` можно подавать как `input=` в `run_experiment_matrix.py`.

### Error analysis и gradient conflict

Для статьи теперь есть отдельные diagnostic scripts:

```bash
PYTHONPATH=src .venv314/bin/python scripts/build_error_analysis_report.py --help
PYTHONPATH=src .venv314/bin/python scripts/probe_gradient_conflict.py --help
```

Первый script даёт slice-level error-rate comparison и конкретные примеры расхождений `multitask vs single-task`. Второй даёт прямой probe по cosine similarity градиентов sentiment/authenticity на shared encoder.

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
