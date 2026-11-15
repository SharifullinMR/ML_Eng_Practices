"""Загрузка датасета в data/raw/."""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
RAW_DIR = Path("data") / "raw"
CSV_NAME = "winequality-red.csv"
EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]
EXPECTED_ROWS = 1599


# скачка
def download_and_extract() -> Path:
    """Скачиваем zip и распаковываем CSV с красным вином"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DIR / CSV_NAME
    zip_path = RAW_DIR / "wine_quality.zip"

    print(f"Скачивается датасет: {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(CSV_NAME, RAW_DIR)
    zip_path.unlink()
    return target_path


def validate(path: Path) -> pd.DataFrame:
    """Проверка существования датасета и, что он нужной формы, нужные имена колонок"""
    if not path.is_file():
        raise FileNotFoundError(f"Ожидался файл {path}, но он не создан")
    df = pd.read_csv(path, sep=";")
    if df.shape != (EXPECTED_ROWS, len(EXPECTED_COLUMNS)):
        raise ValueError(
            f"Ожидалось {EXPECTED_ROWS} строк и {len(EXPECTED_COLUMNS)} колонок, "
            f"получено: {df.shape[0]} строк и {df.shape[1]} колонок"
        )
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Названия колонок не совпадают с ожидаемыми")
    return df


# основной запуск - на скачку, проверка и готовность
def main() -> None:
    path = download_and_extract()
    df = validate(path)
    print(f"Готово: {path} ({df.shape[0]} строк и {df.shape[1]} колонок)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ОШИБКА загрузки данных: {exc}", file=sys.stderr)
        sys.exit(1)
