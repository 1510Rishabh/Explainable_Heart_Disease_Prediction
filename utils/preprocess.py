"""
Preprocessing for the heart disease dataset: imputation, scaling, and consistent feature order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cleveland / common heart dataset column order (must match training CSV)
FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"


def load_raw_data(
    csv_path: Optional[Path] = None,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load heart disease data. Prefer local file; columns are normalized to FEATURE_COLUMNS + target.
    Handles UCI Cleveland format (no header, ? as missing) and standard CSV with column names.
    """
    read_kw = {"na_values": ["?", ""]}

    if csv_path is not None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path, **read_kw)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if not set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns):
                raise ValueError("missing named columns")
        except Exception:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=FEATURE_COLUMNS + [TARGET_COLUMN],
                **read_kw,
            )
            df.columns = [str(c).strip().lower() for c in df.columns]
    elif url:
        df = pd.read_csv(url, **read_kw)
    else:
        raise ValueError("Either csv_path or url must be provided")

    df.columns = [c.strip().lower() for c in df.columns]

    # Map common alternate names
    rename_map = {
        "restingbp": "trestbps",
        "resting_bp": "trestbps",
        "maxhr": "thalach",
        "max_heart_rate": "thalach",
        "exerciseangina": "exang",
        "exercise_angina": "exang",
        "st_depression": "oldpeak",
        "stdepression": "oldpeak",
        "vessels": "ca",
        "num_major_vessels": "ca",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    for c in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Binary target: 1 = disease, 0 = no disease (Cleveland uses 0–4; Kaggle CSV is often 0/1)
    t = df[TARGET_COLUMN].astype(float)
    df[TARGET_COLUMN] = (t > 0).astype(int)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(subset=[TARGET_COLUMN])


def build_preprocessor() -> Pipeline:
    """Impute missing values then scale (helps LR, SVM, KNN; trees are unaffected)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def fit_preprocess(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Fit preprocessor on features; return preprocessor, X, y."""
    cols = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    X = df[cols].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(int)
    prep = build_preprocessor()
    X_t = prep.fit_transform(X)
    return prep, X_t, y


def transform_features(
    preprocessor: Pipeline,
    row: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Single or batch transform; preserves feature order."""
    cols = list(feature_columns) if feature_columns is not None else FEATURE_COLUMNS
    X = row[cols].values.astype(float)
    return preprocessor.transform(X)


def save_preprocessor(preprocessor: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> Pipeline:
    return joblib.load(path)
