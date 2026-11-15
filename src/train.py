"""Обучение модели на вине и сохранение артефактов в artifacts"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data") / "raw" / "winequality-red.csv"
ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2
GOOD_QUALITY_THRESHOLD = 6


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Читаем датасет: 'хорошее вино с качеством (quality >= 6) - положительный класс.
    Классификация бинарная."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"{DATA_PATH} не найден. Сначала выполнить: uv run python -m src.get_data"
        )
    df = pd.read_csv(DATA_PATH, sep=";")
    y = (df["quality"] >= GOOD_QUALITY_THRESHOLD).astype(int)
    x = df.drop(columns=["quality"])
    return x, y


# разделение со стратификацией, обучение, метрики и сохранение
def main() -> None:
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")

    run_report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATA_PATH),
        "n_rows": len(x),
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics": metrics,
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(run_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Модель сохранена: {ARTIFACTS_DIR / 'model.pkl'}")
    print(f"Метрики сохранены: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
