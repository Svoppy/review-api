# Разработка мультитаск-трансформерной модели и веб-системы для совместного анализа тональности и достоверности отзывов в электронной коммерции

## Краткая пометка

Этот текст представляет собой качественный черновик статьи на основе текущего состояния проекта `ReviewGuard`. Данные, датасеты и источники актуализированы по состоянию на `2026-05-26`. Итоговые численные результаты в статье следует заполнить после полного прогона экспериментов на локально загруженных реальных корпусах.

## Аннотация

В статье рассматривается задача совместного анализа тональности и достоверности пользовательских отзывов в электронной коммерции. Актуальность работы определяется тем, что современные e-commerce платформы зависят не только от общей эмоциональной оценки отзывов, но и от их подлинности: даже высокоточная sentiment-модель может быть практически бесполезной, если значимая доля входных отзывов является манипулятивной, заказной или автоматически сгенерированной. На практике эти две задачи чаще всего решаются раздельно, что повышает вычислительные затраты, усложняет сопровождение системы и не позволяет использовать потенциальный положительный перенос знаний между взаимосвязанными признаками отзывов.

Предлагается мультитаск-подход на основе общей трансформерной encoder-архитектуры с двумя классификационными головами: для определения тональности отзыва и для определения его достоверности. Полный диссертационный стек данных включает русскоязычные e-commerce отзывы (`RuReviews`, `Perekrestok Reviews`), классический корпус обманных отзывов `OpSpam`, benchmark `FraudDataset (Yelp)` и мультиязычный набор `MAiDE-up` для AI-generated fake reviews. Однако текущий завершённый pilot уже опирается только на локально подготовленные подвыборки из `RuReviews` и `MAiDE-up`; `Perekrestok Reviews` подготовлен для следующего этапа, `OpSpam` пока не подготовлен локально в этом workspace, а `FraudDataset (Yelp)` остаётся planned extension. Для работы с неоднородной разметкой предлагается унифицированная схема данных, поддерживающая частично размеченные записи.

Практическая значимость исследования состоит в построении воспроизводимого контура `data -> training -> export -> API -> web UI`. В архитектуре веб-системы предусмотрен не только вывод классов и вероятностей, но и lightweight transparency-слой, позволяющий отображать распределения вероятностей по задачам и информацию об усечении входного текста. Этот слой трактуется как инженерная особенность системы, а не как самостоятельный научный вклад. На текущем pilot-этапе мультитаск-модель показывает наиболее убедительный выигрыш именно по authenticity detection, тогда как по sentiment результаты остаются условными и зависят от baseline.

**Ключевые слова:** анализ тональности, подлинность отзывов, fake review detection, multitask learning, Transformer, e-commerce, XLM-RoBERTa, веб-сервис.

## 1. Введение

Отзывы пользователей являются одним из ключевых источников информации в цифровой торговле. Они влияют на репутацию продавца, ранжирование товаров, пользовательское доверие и итоговую конверсию. По этой причине автоматический анализ отзывов стал стандартной задачей прикладной обработки естественного языка. Наиболее распространённое направление в этой области связано с sentiment analysis, то есть определением эмоциональной полярности текста: положительной, нейтральной или отрицательной.

Однако для реальных e-commerce систем анализа одной только тональности недостаточно. Платформы и покупатели сталкиваются с большим количеством недостоверных отзывов: заказных, манипулятивных, спамных и, в последние годы, автоматически сгенерированных большими языковыми моделями. Если такие тексты не фильтруются, то аналитическая система может уверенно извлекать sentiment-сигнал из текста, который не отражает реальный пользовательский опыт. Следовательно, задачи оценки тональности и проверки подлинности отзыва связаны между собой не только прикладным контекстом, но и на уровне признакового пространства.

Несмотря на это, в большинстве прикладных систем sentiment analysis и authenticity detection строятся как отдельные пайплайны. Такой подход понятен с инженерной точки зрения, но он приводит к дублированию инфраструктуры, увеличению стоимости инференса и потере потенциального положительного transfer effect между задачами. С точки зрения исследования это также затрудняет проверку гипотезы о том, может ли совместное обучение улучшить качество одной или обеих задач.

Целью настоящей работы является разработка мультитаск-трансформерной модели и веб-системы, предназначенной для совместной классификации тональности и достоверности отзывов в электронной коммерции. Работа сочетает две плоскости: исследовательскую и инженерную. С одной стороны, предлагается и обосновывается архитектура совместного обучения; с другой стороны, создаётся воспроизводимый прикладной контур, охватывающий нормализацию данных, обучение, экспорт модели и предоставление результатов через API и веб-интерфейс.

## 2. Обзор литературы

Литературу, непосредственно связанную с настоящей работой, целесообразно разделить на четыре пересекающихся направления: multitask learning в NLP, исследования достоверности отзывов, sentiment analysis на реалистичных review-корпусах и explainability для moderation systems. Настоящее исследование располагается именно на пересечении этих направлений, а не внутри какого-то одного из них.

Первое направление — multitask learning как таковое. В классической работе Caruana было показано, что совместное обучение связанных задач может улучшать обобщающую способность модели за счёт общего индуктивного смещения [1]. С распространением pre-trained Transformers эта идея получила новое практическое воплощение. BERT стал универсальной encoder-архитектурой для широкого круга задач классификации текста [2], а MT-DNN продемонстрировал, что схема `shared encoder + task-specific heads` остаётся работоспособной и эффективной в режиме multi-task fine-tuning [3]. При этом современные обзоры MTL в NLP подчёркивают важное ограничение: положительный перенос не возникает автоматически из одного лишь совместного обучения, а зависит от близости задач, плотности supervision signal, способа разделения параметров и динамики оптимизации [4]. Для настоящей статьи это принципиально, поскольку работа не исходит из предположения, что совместное обучение обязано одинаково помочь и sentiment, и authenticity.

Второе направление связано с review authenticity detection. Фундаментальной точкой отсчёта здесь остаётся корпус Ott et al., где была формализована задача deceptive opinion spam detection и предложен gold-standard benchmark truthful/deceptive classification [5]. Позднее исследования начали выходить за рамки одного домена и одного закрытого сценария. Hai et al. показали, что deceptive review detection можно рассматривать как multi-task problem across domains и что relatedness between tasks вместе с unlabeled data помогает в условиях ограниченной разметки [6]. Это особенно важно для настоящей работы, так как её постановка тоже строится на фрагментированных по источникам и меткам корпусах. Дальнейшее развитие authenticity research показало, что поддельность отзыва не исчерпывается поверхностной лингвистикой. Daryani и Caverlee описали задачу hijacked reviews, когда текст выглядит правдоподобно, но семантически не соответствует текущему товарному контексту [7]. В промышленной плоскости Nayak и Garera продемонстрировали ценность unified BERT moderation model для e-commerce, где важны не только метрики, но и единый operational inference contour [8]. В совокупности эти работы подводят к пониманию authenticity как семейства trust-related review problems, а не как одного стабильного бинарного benchmark.

Третье направление — AI-generated reviews. Появление больших языковых моделей заметно изменило сам threat model: fluent synthetic reviews перестали быть редким артефактом и стали масштабируемым механизмом манипуляции. В 2024 году Liyanage et al. показали, что LLM-generated hotel reviews образуют самостоятельную detection-задачу, а не тривиальное продолжение классического deceptive spam setting [9]. В 2025 году корпус `MAiDE-up` дополнительно закрепил важность multilingual evaluation и показал, что сложность detection варьируется по языкам, доменам и sentiment strata [10]. Работы, связанные с `DetectAIRev`, ещё сильнее акцентировали проблему слабой cross-generator и cross-domain robustness даже там, где within-benchmark результаты выглядят высокими [11]. Для настоящей статьи это означает, что authenticity supervision неоднородна не только по датасетам, но и по самому типу негативного примера.

Четвёртое направление относится к sentiment analysis на реалистичных review-корпусах. Для задач e-commerce важны не только узкие benchmark datasets, но и большие естественные коллекции отзывов. Масштабным источником является `Amazon Reviews 2023` [12], а `MARC` подчёркивает значение multilingual sentiment evaluation на review-данных [15]. В русскоязычном сегменте практический интерес представляют `RuReviews` как benchmark продуктовых отзывов [13] и `Perekrestok Reviews` как большой in-domain retail corpus [14]. Именно эти ресурсы особенно хорошо показывают структурную асимметрию публичных данных: sentiment supervision сравнительно легко получить в больших объёмах, тогда как authenticity labels встречаются значительно реже, чаще оказываются шумными и сильнее зависят от конкретной operational definition.

Сквозной прикладной темой для всех четырёх направлений остаётся explainability. Для moderation и trust-oriented systems одной только бинарной метки обычно недостаточно: аналитикам важны confidence patterns, ranked probabilities и сигналы о ненадёжности решения. Современные работы по explainable fake review detection подчеркивают, что даже облегчённый transparency layer может быть практически полезен, если он не выдаётся за полноценную causal interpretability framework [16]. Именно в таком ограниченном и инженерно честном смысле explainability-layer используется и в настоящем репозиторном пакете.

Таким образом, литература указывает на более конкретный research gap, чем просто ``нужна ещё одна мультитаск-модель''. Shared-encoder multitask architectures уже хорошо изучены. Deceptive review detection, включая transfer-aware formulations, тоже исследована. Production moderation systems уже существуют. Недостаточно изученной остаётся именно комбинация этих элементов в одной review-level постановке при partial supervision, где sentiment и authenticity labels приходят из разных публичных корпусов, authenticity охватывает неидентичные benchmark semantics, а все научные claims должны жёстко ограничиваться реально выполненным протоколом. Настоящая статья позиционируется как воспроизводимое исследование именно этой постановки.

## 3. Постановка задачи и исследовательская гипотеза

Пусть задан текстовый отзыв пользователя `x`. Необходимо предсказать два связанных целевых признака:

1. `y_s` — класс тональности, где `y_s ∈ {negative, neutral, positive}`;
2. `y_a` — класс достоверности, где `y_a ∈ {authentic, fake}`.

При этом значимая практическая сложность состоит в том, что доступные публичные корпуса обычно размечают только одну из этих задач. Следовательно, проект должен поддерживать частично размеченные записи, где для одного и того же текста может быть известна только тональность или только authenticity label.

Основная исследовательская гипотеза состоит в том, что совместное обучение на общей трансформерной encoder-части:

- не ухудшит качество sentiment classification по сравнению с single-task Transformer baseline;
- улучшит устойчивость authenticity detection за счёт richer shared language representation;
- позволит построить более компактный и practically deployable web service.

## 4. Научная новизна и вклад работы

Научная новизна исследования заключается в следующем.

1. Предлагается единая мультитаск-постановка для совместного определения тональности и подлинности отзывов в домене электронной коммерции.
2. Формируется унифицированная схема данных, позволяющая объединять частично размеченные корпуса разных источников в единый training format.
3. Формируется расширяемая authenticity-схема, в которую в полном benchmark должны войти и классические deceptive review corpora, и современные AI-generated fake review datasets.
4. Исследование соединяет алгоритмический и системный вклад: от модели и метрик до API и web interface, не выдавая transparency-слой за отдельный научный результат.

С точки зрения позиционирования в литературе работа закрывает разрыв между двумя традиционно раздельными направлениями: `review sentiment modeling` и `review authenticity detection`. Если формулировать это кратко, то статья предлагает shared-encoder multitask architecture + partially labeled heterogeneous corpus design + deployable end-to-end workflow for trust-aware review analysis. При этом цель состоит не в том, чтобы объявить текущий pilot окончательным решением benchmark-level задачи, а в том, чтобы задать воспроизводимую и эмпирически проверяемую схему исследования, показывающую, при каких условиях совместное обучение помогает одной задаче сильнее, чем другой.

Практический вклад работы выражается в разработке системы `ReviewGuard`, реализующей reproducible pipeline:

`raw datasets -> normalized unified records -> baseline/single-task/multitask training -> exported checkpoint -> FastAPI inference -> web interface`.

## 5. Материалы и данные

Для честной постановки экспериментов выбраны реальные и общедоступные датасеты, различающиеся по языку, домену и типу разметки.

| Датасет | Назначение | Язык | Тип меток | Роль в статье |
|---|---|---:|---|---|
| `RuReviews` | Sentiment analysis | RU | `negative/neutral/positive` | основной русскоязычный benchmark |
| `Perekrestok Reviews` | Sentiment analysis | RU | rating-derived sentiment | in-domain retail corpus |
| `FraudDataset (Yelp)` | Authenticity detection | EN | silver fraud labels | planned benchmark, пока не интегрирован локально |
| `OpSpam` | Authenticity detection | EN | truthful/deceptive | planned text-only benchmark, пока нет локальной подготовки |
| `MAiDE-up` | AI-generated fake review detection | multilingual | real vs AI-generated | интегрирован локально и использован в pilot |
| `Amazon Reviews 2023` | Auxiliary sentiment / transfer | EN | rating-derived sentiment | масштабный дополнительный источник |

Выбор именно такой комбинации объясняется тем, что в публичном поле пока отсутствует один крупный и общепризнанный датасет, который одновременно содержал бы полную e-commerce sentiment разметку и высоконадёжную authenticity annotation. Поэтому научно корректнее не маскировать эту проблему, а явно строить unified multitask corpus из нескольких репрезентативных источников. При этом текущая эмпирическая статья уже должна жёстко ограничивать scope завершёнными данными: в текущем workspace локально подготовлены `RuReviews`, `Perekrestok Reviews` и `MAiDE-up`, а reported pilot использует только `RuReviews` и `MAiDE-up`.

## 6. Метод

### 6.1. Общая архитектура

Предлагаемая модель использует hard parameter sharing. Базовый encoder строится на основе `XLM-RoBERTa`, что позволяет работать как с русскоязычными, так и с англоязычными отзывами в рамках единого пространства представлений. После encoder-части применяется pooling, а затем два независимых classification heads:

- `Head_s` для sentiment classification;
- `Head_a` для authenticity detection.

Формально:

`h = Encoder(x)`

`p_s = softmax(W_s h + b_s)`

`p_a = softmax(W_a h + b_a)`

Функция потерь строится как взвешенная сумма:

`L = λ_s * L_sentiment + λ_a * L_authenticity`

где при отсутствии метки по одной из задач соответствующий член loss не учитывается. Это делает модель пригодной для partial supervision.

### 6.2. Базовые модели для сравнения

Для корректной оценки полезности мультитаск-обучения необходимы несколько baseline lines:

1. `TF-IDF + Logistic Regression` для каждой задачи по отдельности;
2. `Single-task Transformer` для sentiment classification;
3. `Single-task Transformer` для authenticity detection;
4. `Multitask Transformer` с общей encoder-частью.

Такой набор базовых моделей позволяет отличить собственно эффект трансформеров от эффекта совместного обучения.

### 6.3. Унифицированная схема данных

В проекте используется единый формат записи:

- `text`
- `sentiment_label`
- `authenticity_label`
- `source`
- `language`
- `domain`
- `metadata`

Подход важен по двум причинам. Во-первых, он позволяет агрегировать разнородные источники. Во-вторых, он делает экспериментальный контур воспроизводимым и пригодным для повторного использования в других доменах.

## 7. Архитектура веб-системы

Разрабатываемая система не ограничивается исследовательской моделью. Для практической применимости реализуется веб-контур, включающий:

1. модуль нормализации и объединения данных;
2. training CLI для classical baseline, single-task Transformer и multitask Transformer;
3. слой экспорта модели;
4. `FastAPI` backend;
5. web UI для интерактивного анализа отзывов.

В API предусмотрен endpoint `/analyze`, который возвращает:

- предсказанный sentiment class;
- confidence score для sentiment;
- предсказанный authenticity class;
- confidence score для authenticity;
- explanation block.

Explainability-слой intentionally сделан lightweight и прозрачным. Он содержит:

- top-ranked probabilities for both tasks;
- token count;
- max sequence length;
- truncation flag;
- explanatory notes for interpretation.

Для прикладной e-commerce системы это важно, поскольку аналитик или исследователь должен понимать не только решение модели, но и степень её уверенности.

## 8. Дизайн экспериментов

### 8.1. Экспериментальные сценарии

Предлагается следующий порядок экспериментов:

1. `RuReviews` + classical baseline для sentiment;
2. `OpSpam` + classical baseline для authenticity;
3. single-task Transformer на sentiment и authenticity по отдельности;
4. мультитаск-обучение на объединённом partially labeled corpus;
5. устойчивость на смешанных authenticity benchmarks (`FraudDataset`, `MAiDE-up`);
6. domain transfer через `Perekrestok Reviews`.

### 8.2. Метрики

Основными метриками выступают:

- `Accuracy`
- `Macro-F1`
- `Weighted-F1`
- `Precision_macro`
- `Recall_macro`
- `Confusion Matrix`

Для authenticity detection при наличии дисбаланса классов особое внимание следует уделить именно `Macro-F1`, а не только accuracy.

### 8.3. Проверяемые гипотезы

В статье целесообразно формализовать следующие гипотезы:

- `H1`: мультитаск-модель сохраняет конкурентное качество по sentiment analysis;
- `H2`: мультитаск-модель показывает более устойчивое качество по authenticity detection, чем text-only baselines;
- `H3`: мультитаск-обучение может давать положительный transfer в условиях mixed corpora и partial supervision, но этот эффект зависит от задачи.

## 9. Ограничения и угрозы валидности

Работа должна честно зафиксировать ограничения.

Во-первых, публичные authenticity datasets различаются по природе ground truth. В `OpSpam` метки ближе к controlled deceptive setting, в `FraudDataset` — к platform-driven silver labels, а в `MAiDE-up` — к AI-generated reviews. Это не один и тот же феномен, и результаты по таким корпусам нельзя интерпретировать как полностью взаимозаменяемые.

Во-вторых, русскоязычный e-commerce сегмент лучше покрыт sentiment data, чем authenticity data. Поэтому для одной из задач мультитаск-модель может получать больше сигналов на английских данных, чем на русских.

В-третьих, explainability-слой в текущем проекте является probabilistic transparency layer, а не полноценной causal interpretability system. Он повышает прозрачность использования, но не должен рассматриваться как научный вклад в explainability research.

## 10. Обсуждение

С научной точки зрения наиболее интересным результатом оказывается не просто максимум F1 на одном корпусе, а качественный анализ transfer dynamics between tasks. Уже текущий pilot показывает asymmetric transfer: shared encoder помогает authenticity detection, но не даёт убедимого выигрыша по sentiment и остаётся слабее classical baseline.

С инженерной точки зрения важен и другой результат: единый encoder уменьшает стоимость инференса и упрощает сопровождение сервиса по сравнению с двумя независимыми моделями. Для e-commerce платформы, где обработка отзывов должна быть дешёвой, воспроизводимой и удобной для интеграции, это является практическим преимуществом.

При этом особенно перспективным направлением остаётся сценарий AI-generated reviews. В 2026 году подлинность текста всё чаще связана не только с человеческим спамом, но и с машинной генерацией, поэтому authenticity detection должна учитывать both human deceptive behavior and synthetic text generation.

## 11. Заключение

В работе предложена концепция и проектная реализация мультитаск-системы для совместного анализа тональности и достоверности отзывов в электронной коммерции. В отличие от традиционного раздельного подхода, система использует общую трансформерную encoder-часть с двумя task-specific heads и поддерживает частично размеченные объединённые корпуса.

Научная ценность работы состоит в соединении нескольких направлений: sentiment analysis, fake review detection, multilingual transformer modeling и system-level reproducibility. Практическая ценность выражается в построении законченного контура от нормализации данных до API и web UI.

Следующий этап исследования заключается уже не в доказательстве работоспособности pipeline, а в расширении benchmark: подключении дополнительных authenticity-корпусов, source-controlled ablation, более крупных выборках и полном dataset-level reporting. После этого данный черновик может быть доведён до полноценной статьи для университетской публикации или использован как основа главы диссертации.

## Сильная формулировка вклада для аннотации и введения

Если нужен более короткий и “публикационный” вариант вклада, его можно формулировать так:

> В статье предлагается воспроизводимый мультитаск-подход к совместному определению тональности и достоверности отзывов в электронной коммерции, объединяющий трансформерную модель, унифицированный partially labeled dataset format и веб-систему с lightweight transparency-слоем как инженерной особенностью.

## Что ещё нужно добавить перед подачей статьи

1. итоговую таблицу метрик на всех корпусах;
2. таблицу сравнения `baseline / single-task / multitask`;
3. абляцию по `loss weights`;
4. error analysis для false positive и false negative cases;
5. иллюстрацию архитектуры системы;
6. финальное оформление литературы под стандарт вуза или журнала.

## Список литературы

1. Caruana R. Multitask Learning. Machine Learning, 1997. URL: <https://link.springer.com/article/10.1023/A:1007379606734>
2. Devlin J., Chang M.-W., Lee K., Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, 2019. URL: <https://aclanthology.org/N19-1423/>
3. Liu X., He P., Chen W., Gao J. Multi-Task Deep Neural Networks for Natural Language Understanding. ACL, 2019. URL: <https://aclanthology.org/P19-1441/>
4. Zhang Y., Yang Q. A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods. arXiv, 2022. URL: <https://arxiv.org/abs/2204.03508>
5. Ott M., Choi Y., Cardie C., Hancock J. T. Finding Deceptive Opinion Spam by Any Stretch of the Imagination. ACL, 2011. URL: <https://aclanthology.org/P11-1032/>
6. Hai Z., Zhao P., Cheng P., Yang P., Li X.-L. Deceptive Review Spam Detection via Exploiting Task Relatedness and Unlabeled Data. EMNLP, 2016. URL: <https://aclanthology.org/D16-1187/>
7. Daryani S., Caverlee J. Identifying Hijacked Reviews. ECNLP, 2021. URL: <https://aclanthology.org/2021.ecnlp-1.9/>
8. Nayak A., Garera N. Deploying Unified BERT Moderation Model for E-Commerce Reviews. EMNLP Industry, 2022. URL: <https://aclanthology.org/2022.emnlp-industry.55/>
9. Liyanage A. et al. Detecting AI-enhanced Opinion Spambots: A Study on LLM-generated Hotel Reviews. ECNLP, 2024. URL: <https://aclanthology.org/2024.ecnlp-1.8/>
10. Ignat O. et al. MAiDE-up: Multilingual Deception Detection of AI-generated Hotel Reviews. Findings of NAACL, 2025. URL: <https://aclanthology.org/2025.findings-naacl.88/>
11. Agrahari N. et al. ProtoFewRoBERTa and DetectAIRev. Findings of IJCNLP-AACL, 2025. URL: <https://aclanthology.org/2025.findings-ijcnlp.132/>
12. Hou Y. et al. Amazon Reviews 2023. McAuley Lab, 2023. URL: <https://amazon-reviews-2023.github.io/main.html>
13. Sismetanin A. RuReviews: An Automatically Annotated Sentiment Analysis Dataset for Product Reviews in Russian. URL: <https://github.com/sismetanin/rureviews>
14. Hugging Face. lapki/perekrestok-reviews dataset card. URL: <https://huggingface.co/datasets/lapki/perekrestok-reviews>
15. Keung P. et al. The Multilingual Amazon Reviews Corpus. EMNLP, 2020. URL: <https://aclanthology.org/2020.emnlp-main.369/>
16. What Matters in Explanations: Towards Explainable Fake Review Detection Focusing on Transformers. arXiv, 2024. URL: <https://arxiv.org/abs/2407.21056>
