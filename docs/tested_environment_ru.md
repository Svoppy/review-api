# Зафиксированная тестовая среда

Дата проверки: `2026-06-11`

## 1. Среда текущего рабочего snapshot

Фактически проверенная локальная среда для текущего desktop snapshot:

- Python: `3.14.2`
- virtualenv: `.venv314`
- shell: `zsh`
- ОС: macOS desktop environment

Именно в этой среде подтверждены:

- полный локальный `pytest`;
- сборка `joint_reviews.current.jsonl`;
- сборка `joint_reviews.balanced6k.jsonl`;
- generation `reports/audit/joint_reviews.balanced6k.audit.json`.

## 2. Pinned environment files

В репозитории также зафиксированы:

- `.python-version` = `3.12.7`
- `requirements-dev.txt`

Это следует трактовать как target pinned environment для более стабильной воспроизводимости, но не как точное описание всех уже сохранённых empirical artifacts.

## 3. Минимальный сценарий установки

Рекомендуемый вариант для текущего snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

Если работа ведётся в уже существующей `.venv314`, для локальных проверок можно использовать:

```bash
source .venv314/bin/activate
python -m pytest -q
```

## 4. Статус

Для article-ready submission важно дополнительно выровнять:

1. environment of record для уже сохранённых `models/` и `reports/`;
2. pinned Python version и реально использованную среду прогонов;
3. runbook-команды и локальные пути к raw/prepared datasets.
