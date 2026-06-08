# Сборка английской LaTeX-статьи

Основной файл:

- `docs/article_final_en.tex`

Библиография:

- `docs/article_final_en.bib`

## Рекомендуемая сборка

```bash
pdflatex article_final_en.tex
bibtex article_final_en
pdflatex article_final_en.tex
pdflatex article_final_en.tex
```

Если используется `latexmk`:

```bash
latexmk -pdf article_final_en.tex
```

## Что важно перед финальной подачей

1. Проверить конкретный шаблон журнала или вуза.
2. Добавить авторов, аффилиации и служебные данные.
3. При необходимости перейти с встроенной библиографии на `\bibliographystyle` и `\bibliography`.
4. Обновить раздел результатов после завершения `single-task` и `multitask` экспериментов.
