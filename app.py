"""
Explainable Heart Disease Prediction — Streamlit UI with SHAP (Random Forest).
Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import predict as pred  # noqa: E402
from utils.explain import (  # noqa: E402
    global_summary_plot,
    individual_waterfall_plot,
    narrative_summary,
    plain_english_bullets,
)
MODEL_DIR = ROOT / "saved_models"


def _ensure_models() -> None:
    if (MODEL_DIR / "random_forest.joblib").exists():
        return
    with st.spinner("First-time setup: downloading data and training models (may take a minute)..."):
        from model.train_model import train

        train()
    pred.clear_caches()


def _feature_inputs(meta: dict) -> dict:
    fr = meta.get("feature_ranges", {})
    row: dict = {}

    st.sidebar.header("Patient inputs")

    lo, hi = fr.get("age", (29.0, 77.0))
    row["age"] = float(st.sidebar.slider("Age", float(lo), float(hi), 55.0, 1.0))

    sex_opt = {"Female (0)": 0, "Male (1)": 1}
    row["sex"] = float(sex_opt[st.sidebar.selectbox("Sex", list(sex_opt.keys()))])

    cp_lo, cp_hi = int(fr.get("cp", (0.0, 3.0))[0]), int(fr.get("cp", (0.0, 3.0))[1])
    row["cp"] = float(
        st.sidebar.selectbox("Chest pain type (cp)", list(range(cp_lo, cp_hi + 1)))
    )

    lo, hi = fr.get("trestbps", (94.0, 200.0))
    row["trestbps"] = float(st.sidebar.slider("Resting BP (trestbps)", lo, hi, 130.0, 1.0))

    lo, hi = fr.get("chol", (126.0, 564.0))
    row["chol"] = float(st.sidebar.slider("Cholesterol (chol)", lo, hi, 240.0, 1.0))

    fbs_opt = {"No (0)": 0, "Yes (1)": 1}
    row["fbs"] = float(fbs_opt[st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", list(fbs_opt.keys()))])

    re_lo, re_hi = int(fr.get("restecg", (0.0, 2.0))[0]), int(fr.get("restecg", (0.0, 2.0))[1])
    row["restecg"] = float(
        st.sidebar.selectbox("Resting ECG", list(range(re_lo, re_hi + 1)))
    )

    lo, hi = fr.get("thalach", (71.0, 202.0))
    row["thalach"] = float(st.sidebar.slider("Max heart rate (thalach)", lo, hi, 150.0, 1.0))

    ex_opt = {"No (0)": 0, "Yes (1)": 1}
    row["exang"] = float(ex_opt[st.sidebar.selectbox("Exercise-induced angina", list(ex_opt.keys()))])

    lo, hi = fr.get("oldpeak", (0.0, 6.2))
    row["oldpeak"] = float(st.sidebar.slider("ST depression (oldpeak)", float(lo), float(hi), 1.0, 0.1))

    sl_lo, sl_hi = int(fr.get("slope", (0.0, 2.0))[0]), int(fr.get("slope", (0.0, 2.0))[1])
    row["slope"] = float(st.sidebar.selectbox("ST slope", list(range(sl_lo, sl_hi + 1))))

    ca_lo, ca_hi = int(fr.get("ca", (0.0, 3.0))[0]), int(fr.get("ca", (0.0, 3.0))[1])
    row["ca"] = float(
        st.sidebar.selectbox("Major vessels colored (ca)", list(range(ca_lo, ca_hi + 1)))
    )

    th_lo, th_hi = int(fr.get("thal", (0.0, 3.0))[0]), int(fr.get("thal", (0.0, 3.0))[1])
    row["thal"] = float(
        st.sidebar.selectbox("Thalassemia (thal)", list(range(th_lo, th_hi + 1)))
    )

    return row


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .risk-high { color: #c0392b; font-weight: 700; font-size: 1.25rem; }
        .risk-low { color: #27ae60; font-weight: 700; font-size: 1.25rem; }
        div.block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _ensure_models()

    meta = pred.load_training_meta()
    accuracies = pred.get_model_accuracies()
    best = pred.get_best_model_name()

    raw_row = _feature_inputs(meta)
    X, proba_map, report = pred.predict_all(raw_row)

    rf_p = proba_map["random_forest"]
    rf_label = report["random_forest"]["label"]

    if "prev_rf_p" not in st.session_state:
        st.session_state.prev_rf_p = rf_p
    delta_p = rf_p - st.session_state.prev_rf_p
    st.session_state.prev_rf_p = rf_p

    st.title("Explainable Heart Disease Prediction")
    st.caption("Multi-model scoring with SHAP explanations (Random Forest, TreeExplainer).")

    # --- Prediction result ---
    st.subheader("Prediction result")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if rf_label == "High risk":
            st.markdown('<p class="risk-high">High risk</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="risk-low">Low risk</p>', unsafe_allow_html=True)
    with c2:
        st.metric("Probability (heart disease)", f"{rf_p:.1%}", delta=f"{delta_p:+.1%}" if delta_p != 0 else None)
    with c3:
        st.info(
            "Green indicates **low** modeled risk; red indicates **high** risk. "
            "Probability is from the Random Forest (primary model for SHAP)."
        )

    st.divider()

    # --- Model comparison ---
    st.subheader("Model comparison")
    rows = []
    for name in ["logistic_regression", "decision_tree", "random_forest", "knn", "svm"]:
        nice = name.replace("_", " ").title()
        rows.append(
            {
                "Model": nice,
                "Accuracy": accuracies[name],
                "Prediction": report[name]["label"],
                "P(disease)": proba_map[name],
            }
        )
    df_cmp = pd.DataFrame(rows)
    t1, t2 = st.columns((1.2, 1))
    with t1:
        st.dataframe(
            df_cmp.style.format({"Accuracy": "{:.3f}", "P(disease)": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"**Best validation accuracy:** {best.replace('_', ' ').title()} ({accuracies[best]:.3f})")
    with t2:
        fig_bar = px.bar(
            df_cmp,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            color_continuous_scale="Blues",
            title="Validation accuracy by model",
        )
        fig_bar.update_layout(showlegend=False, yaxis_range=[0, 1.05], height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Explainability ---
    st.subheader("Explainability (SHAP)")
    st.write(
        "SHAP uses the trained **Random Forest** and **TreeExplainer**. "
        "Click the button to compute plots (summary + waterfall) and plain-language bullets."
    )

    if st.button("Explain prediction", type="primary"):
        st.session_state["show_shap"] = True

    if st.session_state.get("show_shap"):
        rf = pred.load_random_forest()
        X_bg = pred.load_X_background()
        feats: list = meta["feature_columns"]

        with st.spinner("Computing SHAP values..."):
            fig_sum = global_summary_plot(rf, X_bg, feats)
            st.pyplot(fig_sum)
            plt.close(fig_sum)

            fig_w = individual_waterfall_plot(rf, X, feats)
            st.pyplot(fig_w)
            plt.close(fig_w)

        bullets = plain_english_bullets(rf, X, feats)
        st.write(narrative_summary(rf_p))
        for b in bullets:
            st.markdown(f"- {b}")

    st.divider()

    # --- What-if ---
    st.subheader("What-if analysis")
    st.write(
        "All sidebar controls update the prediction **on every change** (full-app what-if). "
        "Adjust age, cholesterol, max heart rate, or any field and watch the metrics above update."
    )
    wf1, wf2, wf3 = st.columns(3)
    with wf1:
        st.metric("Current P(disease) — Random Forest", f"{rf_p:.1%}")
    with wf2:
        st.metric("Current label", rf_label)
    with wf3:
        if abs(delta_p) > 1e-6:
            st.metric("Change vs previous run", f"{delta_p:+.1%}")
        else:
            st.caption("Change a sidebar input to see delta vs previous value.")

    st.divider()
    st.caption("Educational demo — not for clinical use.")


if __name__ == "__main__":
    main()
