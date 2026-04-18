"""
Train and persist heart disease classifiers + preprocessor + SHAP background matrix.
Run from project root: python model/train_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preprocess import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    fit_preprocess,
    load_raw_data,
    save_preprocessor,
)

DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "saved_models"
# UCI Cleveland processed (reliable); GitHub mirrors often move or 404.
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]


def _ensure_dataset() -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH
    import urllib.request

    last_err: Exception | None = None
    for url in DATA_URLS:
        try:
            raw_path = DATA_PATH.with_suffix(".raw.csv")
            urllib.request.urlretrieve(url, raw_path)
            cols = FEATURE_COLUMNS + [TARGET_COLUMN]
            df = pd.read_csv(
                raw_path,
                header=None,
                names=cols,
                na_values=["?", ""],
            )
            df.to_csv(DATA_PATH, index=False)
            raw_path.unlink(missing_ok=True)
            return DATA_PATH
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download dataset. Last error: {last_err}")


def train() -> dict:
    _ensure_dataset()
    df = load_raw_data(csv_path=DATA_PATH)

    preprocessor, X, y = fit_preprocess(df)
    feature_ranges = {
        c: (float(df[c].min()), float(df[c].max())) for c in FEATURE_COLUMNS
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }

    metrics: dict = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test))
        metrics[name] = {"accuracy": acc}
        joblib.dump(clf, MODEL_DIR / f"{name}.joblib")

    best_name = max(metrics, key=lambda k: metrics[k]["accuracy"])

    save_preprocessor(preprocessor, MODEL_DIR / "preprocessor.joblib")

    # Background for SHAP (TreeExplainer): sample of training distribution
    rng = np.random.default_rng(42)
    bg_n = min(300, len(X_train))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    X_background = X_train[bg_idx]
    joblib.dump(X_background, MODEL_DIR / "X_background.joblib")

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "feature_ranges": feature_ranges,
        "metrics": metrics,
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print("Best model:", best_name)
    return meta


if __name__ == "__main__":
    train()
