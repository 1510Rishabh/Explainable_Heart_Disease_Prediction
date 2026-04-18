"""
Load saved models and produce predictions + probabilities for the heart disease app.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_models"

MODEL_FILES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
    "svm",
]


@lru_cache(maxsize=1)
def load_training_meta() -> dict:
    path = MODEL_DIR / "training_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python model/train_model.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_preprocessor():
    path = MODEL_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_random_forest():
    path = MODEL_DIR / "random_forest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_X_background() -> np.ndarray:
    path = MODEL_DIR / "X_background.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return joblib.load(path)


def load_all_models() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in MODEL_FILES:
        p = MODEL_DIR / f"{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        out[name] = joblib.load(p)
    return out


def row_to_dataframe(row: Dict[str, float], feature_columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: row[c] for c in feature_columns}])


def predict_all(
    raw_row: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    raw_row: mapping feature name -> raw value (same scale as CSV).
    Returns: X_processed (1, n_features), proba_high per model, full report with pred label.
    """
    meta = load_training_meta()
    features: List[str] = meta["feature_columns"]
    df = row_to_dataframe(raw_row, features)
    prep = load_preprocessor()
    X = prep.transform(df.values.astype(float))

    models = load_all_models()
    proba_high: Dict[str, float] = {}
    report: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0, 1]
        else:
            p = float(model.predict(X)[0])
        proba_high[name] = p
        pred = 1 if p >= 0.5 else 0
        report[name] = {
            "probability_high_risk": p,
            "prediction": pred,
            "label": "High risk" if pred == 1 else "Low risk",
        }

    return X, proba_high, report


def get_model_accuracies() -> Dict[str, float]:
    meta = load_training_meta()
    return {k: v["accuracy"] for k, v in meta["metrics"].items()}


def get_best_model_name() -> str:
    return load_training_meta()["best_model"]


def clear_caches() -> None:
    load_training_meta.cache_clear()
    load_preprocessor.cache_clear()
    load_random_forest.cache_clear()
    load_X_background.cache_clear()
