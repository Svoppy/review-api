# Сценарий Защиты И Live Demo

## Цель

Этот документ нужен для короткой и уверенной демонстрации проекта `ReviewGuard` на защите диссертации.

## Что показать комиссии за 6-8 минут

1. Проблему: отзывы важны, но одного `sentiment analysis` недостаточно, если часть отзывов является `fake` или `AI-generated`.
2. Научную идею: одна `multitask Transformer` модель решает две задачи одновременно:
   `sentiment` и `authenticity`.
3. Практический результат: модель встроена в рабочее web-приложение с `FastAPI` и браузерным интерфейсом.
4. Честный экспериментальный вывод: на текущем `pilot1k` multitask-модель уже сильнее по `authenticity`, но по `sentiment` еще уступает baseline в сложном mixed-domain режиме.

## Рекомендуемый устный план

### 1. Введение, 40-60 секунд

Скажите примерно так:

> My dissertation focuses on trust-aware review analytics in e-commerce. The core idea is that review polarity alone is not sufficient, because platforms also need to understand whether a review is authentic. Therefore, I developed a multitask Transformer-based model and a web system that jointly predicts sentiment and review authenticity.

### 2. Постановка задачи, 40 секунд

Скажите:

- вход: текст отзыва
- выход 1: `negative / neutral / positive`
- выход 2: `authentic / fake`
- особенность: реальные public datasets обычно размечены только по одной задаче, поэтому использован `partially labeled unified schema`

### 3. Архитектура, 50-70 секунд

Скажите:

- общий multilingual encoder
- две classifier heads
- суммарная multitask loss
- backend: `FastAPI`
- frontend: простой web UI

Если нужен файл с формальной архитектурой:
[Architecture notes](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/architecture.md)

### 4. Эксперименты, 60-90 секунд

Ключевой тезис:

- сравнивались `TF-IDF + Logistic Regression`
- `single-task Transformer`
- `multitask Transformer`

Текущий pilot result:

- `baseline`: sentiment `0.8000 acc / 0.7368 macro-F1`, authenticity `0.8000 / 0.7999`
- `multitask`: sentiment `0.7000 / 0.5885`, authenticity `0.8600 / 0.8580`

Что сказать:

> The current pilot already shows a meaningful result: joint learning improves authenticity detection, while sentiment remains the harder task under mixed-domain and class-sparse conditions.

Подробные цифры:
[Pilot results](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/pilot_results.md)

## Live demo: что запускать

### Шаг 1. Активировать checkpoint

```bash
PYTHONPATH=src .venv314/bin/python scripts/activate_checkpoint.py \
  --source models/pilot1k-multitask \
  --target models/latest
```

### Шаг 2. Быстрая проверка без внешнего сервера

```bash
PYTHONPATH=src .venv314/bin/python scripts/api_smoke.py
```

### Шаг 3. Поднять web-приложение

```bash
PYTHONPATH=src .venv314/bin/uvicorn reviewguard.api.main:app --reload
```

Открыть:
`http://127.0.0.1:8000`

## Live demo: какие примеры показывать

Основные примеры уже подготовлены в:
[data/demo/reviews.json](/Users/diaskazikhanov/Desktop/aitu/nirm/data/demo/reviews.json)

Готовый отчет с фактическими предсказаниями можно пересобрать командой:

```bash
PYTHONPATH=src .venv314/bin/python scripts/generate_defense_demo.py
```

Актуальный отчет:
[Defense demo results](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/defense_demo_results_ru.md)

## Лучший порядок показа

1. `authentic_hotel_en`
   Что подчеркнуть: модель умеет находить реалистичный authentic review.
2. `suspicious_hotel_en`
   Что подчеркнуть: модель уверенно ловит гиперболизированный fake-style отзыв.
3. `ai_style_hotel_en`
   Что подчеркнуть: проект учитывает современную угрозу `AI-generated reviews`.
4. `product_positive_ru`
   Что подчеркнуть: русскоязычный sentiment распознается хорошо.
5. `product_neutral_ru`
   Что подчеркнуть: это честный limitation case, который объясняет текущую слабость `neutral` класса.

## Если live demo пойдет неидеально

Говорите это спокойно:

> The demo uses a real pilot checkpoint, so I intentionally keep the limitations visible. This is not a mocked system. The strong side is authenticity detection, while the main current limitation is sentiment stability on heterogeneous data.

Это не минус, а плюс для защиты: вы показываете научную добросовестность.

## Последняя фраза для завершения

> In summary, the project already delivers a working end-to-end system and a real multitask pilot result. The next step is to scale the benchmark and further improve sentiment robustness without losing the authenticity gains.
