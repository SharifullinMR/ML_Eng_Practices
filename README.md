# ДЗ 1. Воспроизводимый ML-проект.
# Качество вина (Wine Quality)

Учебный воспроизводимый ML-проект (ДЗ-1).

Задача: по химическим показателям предсказать, является ли красное вино «хорошим»
target - (quality >= 6).

## Датасет

- **Источник:** [Machine Learning Repository Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Размер: 1599 строк (красное вино), 12 колонок, 11 признаков и таргет.
- Скрипт `src.get_data` скачивает данные с нуля по прямой ссылке.

## Установка

```bash
git clone https://github.com/SharifullinMR/ml-project.git
cd ml-project
uv sync
```

`uv sync` создаёт виртуальное окружение `.venv` и ставит точные зависимости из `uv.lock`.

## Загрузка данных

```bash
uv run python -m src.get_data
```

Скачиваетcя архив, конечный распакованный файл в  `data/raw/winequality-red.csv`
и проверяется файл (1599 строк, 12 колонок).

## Проверка кода командами

(если нужно run после коммита или в проекте настроен автоматически сам с git commit)

```bash
uv run pre-commit run --all-files
uv run ruff check .
```

## Обучение ML модели и её сохранение

```bash
uv run python -m src.train
```

Сохраненные модели и метрики
- `artifacts/model.pkl`  модель (StandardScaler + LogisticRegression)
- `artifacts/metrics.json`метрики текущего запуска (accuracy, F1, ROC AUC)

## Последовательность команд для проверки проекта:
```bash
cd %USERPROFILE%
git clone https://github.com/USERNAME/ml-project.git ML-project-check
cd ML-project-check
uv sync
uv run python -m src.get_data
uv run python -m src.train
```

### Если все отработало без ошибок, то можно удалить за собой:
```bash
cd %USERPROFILE%
rmdir /s /q ML-project-check
```


## Структура всего проекта

```
ml-project/
├── pyproject.toml          # зависимости и настройки Ruff
├── uv.lock                 # зафиксированные версии
├── .pre-commit-config.yaml # конфиг для pre-commit, запуск Ruff перед коммитом
├── src/
│   ├── get_data.py         # загрузка и валидация датасета
│   └── train.py            # обучение данных и сохранение модели/артефактов
├── data/raw/               # создаётся скриптом, лежат данные, в git не попадает
├── artifacts/              # создаётся скриптом, лежает различные артефакты, в git не попадает
└── reports/hw-1.md         # отчёт по ДЗ-1
```
