# Defense Demo Results

## Active checkpoint

- status: `ok`
- model_ready: `True`
- model_name: `distilbert-base-multilingual-cased`
- checkpoint_dir: `models/latest`

## Summary table

| ID | Language | Sentiment | Sentiment confidence | Authenticity | Authenticity confidence |
|---|---|---:|---:|---:|---:|
| `authentic_hotel_en` | `en` | `negative` | `0.8897` | `authentic` | `0.8058` |
| `authentic_apartment_en` | `en` | `negative` | `0.7478` | `authentic` | `0.6848` |
| `suspicious_hotel_en` | `en` | `positive` | `0.8833` | `fake` | `0.7224` |
| `ai_style_hotel_en` | `en` | `positive` | `0.8131` | `fake` | `0.9204` |
| `product_positive_ru` | `ru` | `positive` | `0.8812` | `fake` | `0.8203` |
| `product_neutral_ru` | `ru` | `positive` | `0.8321` | `fake` | `0.8566` |

## Аутентичный сбалансированный отзыв о проживании

- id: `authentic_hotel_en`
- why_in_demo: Показать, что модель умеет находить правдоподобный, неброский authentic review.
- speaker_note: Это хороший первый пример, потому что он звучит естественно и без рекламных гипербол.
- text: We stayed for one night before an early flight. Check-in was quick, the room was clean enough, and the price matched what we got.
- sentiment: `negative` (`0.8897`)
- authenticity: `authentic` (`0.8058`)
- sentiment_top_probabilities: negative=0.890, positive=0.105, neutral=0.006
- authenticity_top_probabilities: authentic=0.806, fake=0.194
- token_info: `33/128`, truncated=`False`

## Еще один аутентичный отзыв без преувеличений

- id: `authentic_apartment_en`
- why_in_demo: Подтвердить, что модель не обязана видеть только negative/fake паттерны.
- speaker_note: В этом примере есть реалистичный тон: умеренная оценка, без чрезмерной эмоциональности.
- text: The apartment looked exactly like the photos. Nothing luxurious, but everything worked and the host answered messages quickly.
- sentiment: `negative` (`0.7478`)
- authenticity: `authentic` (`0.6848`)
- sentiment_top_probabilities: negative=0.748, positive=0.246, neutral=0.006
- authenticity_top_probabilities: authentic=0.685, fake=0.315
- token_info: `26/128`, truncated=`False`

## Подозрительно восторженный fake-style отзыв

- id: `suspicious_hotel_en`
- why_in_demo: Показать сильную сторону authenticity detection на гиперболизированном отзыве.
- speaker_note: Здесь полезно акцентировать, что текст слишком идеализирован и похож на шаблонный promotional review.
- text: This hotel is absolutely perfect in every possible way, unbelievably luxurious, flawless, magical, and life-changing for everyone.
- sentiment: `positive` (`0.8833`)
- authenticity: `fake` (`0.7224`)
- sentiment_top_probabilities: positive=0.883, negative=0.113, neutral=0.004
- authenticity_top_probabilities: fake=0.722, authentic=0.278
- token_info: `35/128`, truncated=`False`

## AI-like synthetic review

- id: `ai_style_hotel_en`
- why_in_demo: Показать чувствительность модели к современному типу угроз: гладкий LLM-like review.
- speaker_note: Хороший пример для связи с MAiDE-up и угрозой AI-generated reviews.
- text: An exceptional and unforgettable stay with impeccable ambiance, superior comfort, and an extraordinary level of service beyond expectations.
- sentiment: `positive` (`0.8131`)
- authenticity: `fake` (`0.9204`)
- sentiment_top_probabilities: positive=0.813, negative=0.179, neutral=0.008
- authenticity_top_probabilities: fake=0.920, authentic=0.080
- token_info: `32/128`, truncated=`False`

## Русскоязычный положительный отзыв о товаре

- id: `product_positive_ru`
- why_in_demo: Показать, что sentiment по-русски работает, но authenticity на pilot-модели еще нестабильна.
- speaker_note: Это удобно использовать как честный limitation case: тональность распознается хорошо, а authenticity может ошибаться.
- text: Отличный товар, пришел быстро, упаковка аккуратная, качество полностью устроило.
- sentiment: `positive` (`0.8812`)
- authenticity: `fake` (`0.8203`)
- sentiment_top_probabilities: positive=0.881, negative=0.117, neutral=0.002
- authenticity_top_probabilities: fake=0.820, authentic=0.180
- token_info: `29/128`, truncated=`False`

## Русскоязычный близкий к neutral отзыв

- id: `product_neutral_ru`
- why_in_demo: Показать слабое место pilot-конфигурации: редкий neutral класс и смешение доменов.
- speaker_note: Это лучший пример, чтобы объяснить, почему macro-F1 по sentiment пока неидеален.
- text: Обычный товар, характеристики соответствуют описанию, доставка стандартная, без сюрпризов.
- sentiment: `positive` (`0.8321`)
- authenticity: `fake` (`0.8566`)
- sentiment_top_probabilities: positive=0.832, negative=0.164, neutral=0.004
- authenticity_top_probabilities: fake=0.857, authentic=0.143
- token_info: `28/128`, truncated=`False`

## Presenter reminder

- This demo uses the current `pilot1k` multitask checkpoint, so it is a real model artifact rather than a mocked service.
- The strongest practical talking point is authenticity detection on exaggerated or AI-like reviews.
- The most honest limitation to show is weaker sentiment stability on Russian near-neutral product reviews.
