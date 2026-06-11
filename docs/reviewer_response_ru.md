# Ответ На Ключевые Замечания Рецензента

## 1. Масштаб эксперимента критически мал

Замечание принято.

Что уже сделано в тексте:

- в [article_draft_en.md](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_draft_en.md) pilot теперь прямо описан как `pilot-stage evidence`, а не как benchmark;
- добавлено явное ограничение: `2,000` записей, `200` sentiment test, `100` authenticity test, `1 seed`, `1 epoch`;
- все улучшения метрик обозначены как `descriptive`, а не как statistically verified.

Что остается сделать по существу:

- repeated runs
- significance testing
- larger benchmark
- увеличить benchmark subset минимум до нескольких тысяч записей с контролем `neutral`-support
- публиковать split-audit вместе с train report

## 2. Не все заявленные датасеты реально интегрированы

Замечание принято.

Что уже сделано в тексте:

- таблица данных теперь явно разделяет:
  - `used in current pilot`
  - `integrated locally, reserved for expanded benchmark`
  - `planned benchmark, not used in current pilot`
- `OpSpam` и `FraudDataset` больше не выглядят как будто они уже участвуют в завершенных экспериментах.

Что остается сделать по существу:

- локально интегрировать `OpSpam`
- локально интегрировать `FraudDataset`
- прогнать их в полном benchmark

## 3. Только 1 epoch обучения

Замечание принято.

Что уже сделано в тексте:

- статья теперь прямо говорит, что текущий training protocol недостаточен для сильных comparative claims;
- отмечено, что `1 epoch` и `1 seed` это компромисс ради локальной воспроизводимости.

Что остается сделать по существу:

- увеличить число эпох
- ввести checkpoint selection
- провести multiple-seed evaluation
- включить `early stopping` и class balancing как часть базового протокола, а не ad hoc tuning

## 4. Asymmetric transfer описан, но механизм не объяснен

Замечание принято.

Что уже сделано в тексте:

- в Discussion и Error Analysis добавлены три рабочие гипотезы:
  - `gradient conflict`
  - `domain-task entanglement`
  - `label-noise / class-sparsity asymmetry`

Что остается сделать по существу:

- ablation по источникам
- source-controlled experiments
- leave-one-source-out / source-wise robustness tables
- при желании gradient-similarity analysis

## 5. Explanation layer не является научным вкладом

Замечание принято.

Что уже сделано в тексте:

- `explanation layer` переименован в `auxiliary transparency object` / `transparency layer`;
- в статье прямо указано, что это не scientific contribution, а engineering usability feature.

Что остается сделать по существу:

- не строить вокруг этого novelty claim
- вынести в system implementation subsection или appendix без novelty claim
- если оставлять в main text, связывать только с deployment transparency, а не с H1-H3

## 6. Смешение доменов и языков не контролируется

Замечание принято.

Что уже сделано в тексте:

- статья теперь прямо связывает риск с текущим pilot:
  `RuReviews` + `MAiDE-up`
- добавлено, что observed gains could still be affected by source-specific shortcuts
- это вынесено в benchmark extensions и limitations

Что остается сделать по существу:

- dataset-source ablation
- leave-one-source-out evaluation
- дополнительные authenticity corpora, чтобы снять зависимость от одного `MAiDE-up`
- явно операционализировать robustness как `mean slice Macro-F1`, `worst-slice Macro-F1`, `robustness gap`

## 7. H3 была слишком расплывчатой

Замечание принято.

Что должно быть исправлено по существу:

- переписать H3 в измеримый вид;
- фиксировать заранее:
  - source-wise mean Macro-F1,
  - worst-slice Macro-F1,
  - best-minus-worst gap,
  - leave-one-source-out delta;
- считать `joint training improves robustness` подтвержденным только если multitask улучшает хотя бы два из этих показателей на одном и том же протоколе.

## 8. Заголовок и позиционирование завышают масштаб

Замечание принято.

Что следует сделать:

- оставить в названии `Toward` / `pilot study`, если full benchmark ещё не завершён;
- не называть работу benchmark paper, пока не завершены крупные source-controlled эксперименты;
- в аннотации и введении сразу обозначать `pilot-backed empirical study`.

## Итоговая позиция

После правок статья стала более строгой и честной. Сейчас её лучше позиционировать как:

- сильный `pilot study`
- `workshop-scale empirical report`
- качественную главу диссертации

А не как финальный `ACL/EMNLP-level benchmark paper`.
