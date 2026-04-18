"""
SHAP explainability for Random Forest (TreeExplainer) and plain-English summaries.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shap

# Human-readable labels for UI and explanations
DISPLAY_NAMES: Dict[str, str] = {
    "age": "Age",
    "sex": "Sex (1=male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels colored",
    "thal": "Thalassemia",
}


def _tree_shap_positive_class(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    """SHAP values for class 1 (heart disease), shape (n_samples, n_features)."""
    out = explainer.shap_values(X)
    if isinstance(out, list):
        return np.asarray(out[1])
    arr = np.asarray(out)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr


def _expected_value_positive(explainer: shap.TreeExplainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        flat = np.ravel(ev)
        if flat.size > 1:
            return float(flat[1])
    return float(np.ravel(ev)[0])


def global_summary_plot(
    rf_model,
    X_background: np.ndarray,
    feature_names: List[str],
    max_samples: int = 400,
) -> plt.Figure:
    """SHAP summary plot (beeswarm) for global feature importance."""
    rng = np.random.default_rng(42)
    n = min(max_samples, len(X_background))
    idx = rng.choice(len(X_background), size=n, replace=False)
    X_sample = X_background[idx]

    explainer = shap.TreeExplainer(rf_model)
    shap_vals = _tree_shap_positive_class(explainer, X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def individual_waterfall_plot(
    rf_model,
    X_single: np.ndarray,
    feature_names: List[str],
) -> plt.Figure:
    """Waterfall plot for one preprocessed row (shape (1, n_features))."""
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = _tree_shap_positive_class(explainer, X_single)
    row_vals = shap_vals[0]
    base = _expected_value_positive(explainer)
    labels = [DISPLAY_NAMES.get(f, f.replace("_", " ").title()) for f in feature_names]

    explanation = shap.Explanation(
        values=row_vals,
        base_values=base,
        data=X_single[0],
        feature_names=labels,
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=14, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plain_english_bullets(
    rf_model,
    X_single: np.ndarray,
    feature_names: List[str],
    top_k: int = 6,
) -> List[str]:
    """
    Turn SHAP contributions into short bullet strings for the positive class.
    Positive SHAP → pushes prediction toward heart disease (high risk).
    """
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = _tree_shap_positive_class(explainer, X_single)
    row = shap_vals[0]

    order = np.argsort(np.abs(row))[::-1][:top_k]
    bullets: List[str] = []
    for i in order:
        fname = feature_names[i]
        val = float(row[i])
        label = DISPLAY_NAMES.get(fname, fname.replace("_", " ").title())
        if val > 0:
            bullets.append(f"{label} increased the estimated risk in this model.")
        elif val < 0:
            bullets.append(f"{label} decreased the estimated risk in this model.")
        else:
            bullets.append(f"{label} had a neutral effect on this prediction.")

    return bullets


def narrative_summary(proba_high: float) -> str:
    risk = "higher" if proba_high >= 0.5 else "lower"
    return (
        f"The Random Forest assigns a {risk} probability of heart disease (about {proba_high:.0%}). "
        "The bullets below interpret SHAP contributions for this prediction."
    )
