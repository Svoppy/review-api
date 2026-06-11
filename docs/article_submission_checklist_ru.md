# Чеклист доведения статьи до подачи

Дата обновления: `2026-06-11`

## 1. Что уже готово

- есть рабочий end-to-end pipeline `data -> training -> export -> API -> web UI`;
- есть complete pilot package: `baseline`, `single-task`, `multitask`, `3-seed` summary, statistics, task ablation;
- есть leakage-aware split logic и article-grade dataset audit;
- есть локально собранные корпуса `joint_reviews.current.jsonl` и `joint_reviews.balanced6k.jsonl`;
- есть основной русский текст статьи: [article_final_ru.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_final_ru.md).

## 2. Что обязательно закрыть перед подачей

### Эксперименты

- перегенерировать ключевые baseline / single-task / multitask runs под текущим протоколом `configs/model.pilot.yaml`;
- либо сузить article scope до bounded pilot, либо добавить в реальные прогоны хотя бы один не-`MAiDE-up` authenticity benchmark;
- собрать article-facing robustness package по `source`, `domain`, `language`;
- закрыть или снять из narrative ablations `source-balanced vs naive` и `loss_weights`.

### Анализ результатов

- вынести в статью или приложение confusion matrices;
- добавить краткий error analysis;
- держать в актуальном состоянии canonical appendix [article_results_package_ru.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_results_package_ru.md);
- явно показать class support и ограничения по `neutral` / authenticity coverage.

### Воспроизводимость

- синхронизировать `docs/reproducibility_note_ru.md`, `docs/tested_environment_ru.md`, runbook и README с фактическим snapshot;
- зафиксировать один environment of record для article results;
- убрать или переписать команды, которые требуют отсутствующих в snapshot raw paths без явной оговорки.

### Текст статьи

- привести backbone narrative в соответствие с фактически reported pilot results;
- заменить plan-like формулировки на completed-study wording там, где результаты уже есть;
- не оставлять скрытых broad claims про robustness или multitask superiority.

## 3. Что не надо делать

- не подавать текущий pilot как completed multi-benchmark study;
- не выдавать `MAiDE-up`-центричный authenticity result за общий e-commerce authenticity benchmark;
- не писать, что текущий пакет уже валидирует одинаково human deception, silver fraud и AI-generated reviews;
- не называть transparency-layer полноценной explainability system;
- не придумывать финальные цифры без новых прогонов под текущим протоколом.

## 4. Минимальный комплект для сильной подачи

- финальная статья `docs/article_final_ru.md`, синхронизированная с фактическими result artifacts;
- canonical appendix `docs/article_results_package_ru.md`, собранный из реальных report artifacts;
- таблица сравнения `baseline / single-task / multitask`;
- robustness table по `source/domain/language`;
- confusion matrices и краткий error analysis;
- limitations / threats to validity section;
- список воспроизводимых команд для основного experimental path.

## 5. Критерий готовности

Работу можно считать близкой к `submission-ready`, если выполнены все условия:

1. central experimental claims опираются на реальные артефакты в `reports/` и `models/`;
2. статья не обещает больше, чем покрывает текущий benchmark bundle;
3. reproducibility docs совпадают с фактической средой и командами;
4. pilot package либо честно позиционирован как bounded study, либо расширен до более широкого authenticity benchmark;
5. все главные выводы в разделе результатов поддержаны таблицей, метрикой или рисунком в статье или в canonical appendix.
