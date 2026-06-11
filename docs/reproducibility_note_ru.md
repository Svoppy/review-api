# Примечание по воспроизводимости

Дата проверки: `2026-06-11`

## 1. Что воспроизводится уже сейчас

- импорт пакета `reviewguard` и структура модулей;
- data pipeline: нормализация, merge, split, audit;
- вычисление метрик, статистический слой и robustness utilities;
- baseline / single-task / multitask training CLI на уровне команд и конфигов;
- формат экспортируемых артефактов и API wiring;
- текущий набор автоматизированных тестов.

## 2. Что фактически проверено в текущем snapshot

Команда:

```bash
PYTHONPATH=src .venv314/bin/python -m pytest -q
```

Результат:

- `61 passed in 4.58s`

Это означает, что текущий snapshot уже воспроизводим на уровне локальной логики пакета, тестового набора и основных исследовательских utility-слоёв.

## 3. Что всё ещё ограничивает полную воспроизводимость статьи

- reported pilot artifacts были собраны не в том же протоколе, который сейчас описан как минимальный standard в `configs/model.pilot.yaml` и `docs/final_experiment_runbook_ru.md`;
- в локальном workspace отсутствуют подготовленные raw/export corpora для `OpSpam` и `FraudYelpDataset`, поэтому completed empirical scope по authenticity остаётся уже, чем intended final benchmark;
- часть raw-data команд в документации требует локально подготовленных файлов и не должна трактоваться как гарантированно воспроизводимая “с нуля” команда для любого snapshot;
- environment of record для уже сохранённых model artifacts и текущий pinned environment требуют дополнительного выравнивания перед submission-ready claim.

## 4. Корректная формулировка для статьи

На текущем этапе корректно утверждать следующее:

> Репозиторий содержит воспроизводимый исследовательский контур и полный проходящий локальный тестовый пакет для кодовой базы. В то же время полная воспроизводимость именно article-level empirical results требует согласования environment of record, повторного прогона ключевых model runs под текущим протоколом и расширения локально подготовленного benchmark bundle.

## 5. Что нужно сделать для stronger submission claim

1. Перегенерировать ключевые baseline / single-task / multitask runs под текущим протоколом `epochs=4`, `balanced`, `source-balanced`, `early stopping`.
2. Зафиксировать один canonical environment of record для article results.
3. Синхронизировать runbook-команды с реально существующими local paths и prepared datasets.
4. Дополнить локальный authenticity bundle хотя бы одним не-`MAiDE-up` benchmark.
