# Статус усиленного pilot-rerun (`pilot1k_v2`)

Обновлено: `2026-06-12`

## Что это за слой

Этот файл фиксирует статус усиленного rerun по обновлённому протоколу `configs/model.pilot.yaml`:

- `epochs=4`
- balanced class weighting
- source-balanced sampling для multitask
- early stopping
- repeated train seeds

Он нужен, чтобы не смешивать:

1. **завершённый legacy pilot evidence package** в `reports/multiseed/pilot1k/*`
2. **in-progress strengthened rerun** в `models/multiseed/pilot1k_v2/*`

## Текущая степень готовности

Завершено:

- `baseline`: `seed-11`, `seed-21`, `seed-42`
- `single-task-sentiment`: `seed-11`, `seed-21`, `seed-42`
- `single-task-authenticity`: `seed-11`, `seed-21`, `seed-42`

Пока не завершено:

- `multitask`: сейчас присутствует только `seed-11`
- для `pilot1k_v2` ещё нет итоговых `summary.json`, `statistics_summary.json` и финального ablation package

Из-за этого `pilot1k_v2` пока **нельзя** цитировать в статье как completed comparative package.

## Ранние наблюдения

Уже завершённые single-task прогоны показывают, что усиленный протокол заметно меняет картину по сравнению со старым pilot.

### Single-task Sentiment Macro-F1

- `seed-11`: `0.8789`
- `seed-21`: `0.8723`
- `seed-42`: `0.8385`

### Single-task Authenticity Macro-F1

- `seed-11`: `0.9108`
- `seed-21`: `0.9103`
- `seed-42`: `0.9002`

### Multitask Seed-11 Snapshot

- sentiment Macro-F1: `0.8276`
- authenticity Macro-F1: `0.8694`

Эти числа полезны как engineering/protocol evidence: они подтверждают, что новый протокол реально активен и даёт иную динамику, чем legacy pilot. Но этого **недостаточно** для новой article-level интерпретации, потому что multitask sweep ещё не завершён.

## Текущая интерпретация

На сегодня самый честный вывод такой:

- в репозитории уже есть **completed legacy pilot package** и **in-progress stronger rerun path**;
- усиленный rerun уже показывает, что поведение моделей существенно меняется при новом протоколе;
- пока все multitask seeds не завершены и пока не собраны новые aggregate reports, strengthened rerun нужно трактовать как промежуточный validation layer, а не как новый центральный evidence package статьи.

## Что нужно сделать дальше

1. завершить `multitask` по всем target seeds;
2. собрать multi-seed summary, statistical comparison и ablation artifacts для `pilot1k_v2`;
3. сопоставить strengthened rerun с legacy pilot package;
4. только после этого решать, заменяет ли новый пакет основной empirical layer статьи или, наоборот, дополнительно сужает claims.
