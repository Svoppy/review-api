# Структура Слайдов Для Защиты

## Слайд 1. Тема и актуальность

- название диссертации
- почему e-commerce review analysis важен
- почему одной тональности недостаточно

## Слайд 2. Проблема

- `sentiment analysis` не учитывает достоверность отзыва
- `fake` и `AI-generated` reviews искажают бизнес-аналитику
- нужны совместные predictions: `sentiment + authenticity`

## Слайд 3. Цель и задачи

- разработать multitask Transformer-based model
- разработать web system
- сравнить `baseline`, `single-task`, `multitask`

## Слайд 4. Данные

- `RuReviews`
- `Perekrestok Reviews`
- `MAiDE-up`
- дополнительно в полном benchmark: `OpSpam`, `FraudDataset`
- проблема: данные частично размечены

## Слайд 5. Архитектура модели

- shared multilingual encoder
- `sentiment head`
- `authenticity head`
- weighted multitask loss

Материал для схемы:
[Architecture notes](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/architecture.md)

## Слайд 6. Архитектура системы

- data normalization
- training CLI
- export checkpoint
- `FastAPI` backend
- web UI

Можно вставить диаграмму из:
[English article draft](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/article_draft_en.md)

## Слайд 7. Экспериментальная схема

- `TF-IDF + Logistic Regression`
- `single-task Transformer`
- `multitask Transformer`
- единый pilot protocol

## Слайд 8. Основные результаты

- `baseline`: sentiment `0.8000 / 0.7368`, authenticity `0.8000 / 0.7999`
- `multitask`: sentiment `0.7000 / 0.5885`, authenticity `0.8600 / 0.8580`
- главный вывод: multitask уже полезен для `authenticity`

Источник:
[Pilot results](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/pilot_results.md)

## Слайд 9. Ошибки и ограничения

- слабый `neutral` класс
- mixed-domain transfer
- source bias в authenticity
- pilot еще не равен full benchmark

## Слайд 10. Демонстрация приложения

- web UI
- `/analyze`
- 2-3 готовых примера

Источник:
[Defense demo results](/Users/diaskazikhanov/Desktop/aitu/nirm/docs/defense_demo_results_ru.md)

## Слайд 11. Практическая значимость

- модерация отзывов
- trust-aware analytics
- снижение риска искаженной репутации товара/продавца

## Слайд 12. Заключение и future work

- система уже работает end-to-end
- получен реальный pilot result
- следующий шаг: full benchmark и усиление sentiment branch
