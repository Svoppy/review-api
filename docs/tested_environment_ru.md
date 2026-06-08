# Зафиксированная тестовая среда

Текущий репозиторий следует считать ориентированным на следующую базовую среду:

- Python: `3.12.7`
- ОС: Windows desktop environment
- shell examples: PowerShell

## Файлы фиксации среды

- `.python-version`
- `requirements-dev.txt`

## Минимальный сценарий установки

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

## Статус

Эта фиксация среды делает пакет более воспроизводимым на уровне зависимостей и версии Python, но не заменяет необходимость:

1. локально подготовить реальные данные;
2. отдельно сохранить итоговые model artifacts;
3. выполнить полный training and evaluation workflow.
