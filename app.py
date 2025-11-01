from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt

from src.config import MODEL_PATH, MODELS_DIR, TARGET_COL, ID_COLS
from src.data import basic_clean
from src.pipeline import engineer_features

st.set_page_config(page_title="Readmission Risk (UCI Diabetes)", layout="wide")

st.title("Hospital Readmission Prediction")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feat_file = MODELS_DIR / "feature_columns.json"
    feature_cols = None
    if feat_file.exists():
        with open(feat_file, "r", encoding="utf-8") as f:
            feature_cols = json.load(f).get("feature_columns")
    return model, feature_cols


model, feature_cols = load_model()

if model is None:
    st.error("Model not found. Please run training first: python -m src.train")
    st.stop()

st.sidebar.header("Actions")

uploaded = st.file_uploader("Upload CSV (raw UCI schema)", type=["csv"]) 

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    df = basic_clean(df)
    # Align columns
    if feature_cols is not None:
        present = [c for c in feature_cols if c in df.columns]
        X = df[present].copy()
    else:
        # Fallback: drop target/IDs if present
        drop_cols = [c for c in [TARGET_COL] + ID_COLS if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")

    # Ensure missing values are numpy nan
    X = X.where(pd.notna(X), np.nan)

    # Status block to inform user during processing
    with st.status(
        "Processing... Please wait while we generate predictions and insights.",
        expanded=False,
    ) as status:
        status.update(label="Generating predictions...", state="running")

        preds = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[:, 1]
            except Exception:
                proba = None

        out = X.copy()
        out["pred_readmit_30d"] = preds
        if proba is not None:
            out["readmit_prob"] = proba

        st.subheader("Predictions")
        st.dataframe(out.head(50))

        st.download_button(
            label="Download predictions as CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        status.update(label="Computing insights and visualizations...", state="running")

        # --- KPIs & Insights ---
        st.markdown("---")
        st.subheader("Readmission Insights")

        # Overall predicted readmission rate
        rate = float(np.mean(preds)) * 100.0 if len(preds) > 0 else 0.0
        st.write(
            f"Predicted readmission rate: {rate:.1f}% ({int(np.sum(preds))}/{len(preds)})"
        )

        # Donut chart for readmission rate
        try:
            st.markdown("**Readmission rate (donut)**")
            pos = int(np.sum(preds))
            neg = int(len(preds) - pos)
            total = pos + neg
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            if total > 0:
                ax.pie(
                    [pos, neg],
                    labels=["Predicted readmit", "Not readmit"],
                    colors=["#F94144", "#90BE6D"],
                    startangle=90,
                    autopct="%1.1f%%",
                    wedgeprops=dict(width=0.4, edgecolor="white"),
                )
            else:
                ax.pie(
                    [1],
                    labels=["No data"],
                    colors=["#CCCCCC"],
                    startangle=90,
                    wedgeprops=dict(width=0.4, edgecolor="white"),
                )
            ax.set(aspect="equal")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"Donut chart unavailable: {e}")

        # Feature importance (best-effort mapping to feature names)
        st.markdown("**Top feature drivers (model importance)**")
        try:
            # Access underlying classifier and feature names from preprocessing
            clf = None
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                clf = model.named_steps["clf"]
            importances = getattr(clf, "feature_importances_", None)
            feat_names = None
            if hasattr(model, "named_steps") and "pre" in model.named_steps:
                pre = model.named_steps["pre"]
                try:
                    feat_names = pre.get_feature_names_out()
                except Exception:
                    try:
                        feat_names = pre.named_steps["ct"].get_feature_names_out()
                    except Exception:
                        feat_names = None

            if importances is not None:
                importances = np.asarray(importances)
                if feat_names is None or len(feat_names) != importances.shape[0]:
                    feat_names = np.array([f"f{i}" for i in range(importances.shape[0])])
                else:
                    feat_names = np.array(feat_names)

                top_k = min(15, importances.shape[0])
                order = np.argsort(importances)[::-1][:top_k]
                imp_df = pd.DataFrame(
                    {"feature": feat_names[order], "importance": importances[order]}
                )
                imp_df = imp_df.set_index("feature")
                st.bar_chart(imp_df)
            else:
                st.info("Feature importances not available for this model.")
        except Exception as e:
            st.warning(f"Could not compute feature importances: {e}")

        # Suggestions based on engineered cohort signals
        st.markdown("**Suggestions to reduce readmission risk (data-driven heuristics)**")
        try:
            df_eng = engineer_features(df)
            df_eng = df_eng.loc[X.index]
            pos_mask = (preds == 1)

            suggestions = []
            def add(msg):
                if msg not in suggestions:
                    suggestions.append(msg)

            # Prior visits
            if "prior_visits" in df_eng.columns and pos_mask.any():
                pv_pos = float(df_eng.loc[pos_mask, "prior_visits"].mean())
                pv_all = float(df_eng["prior_visits"].mean())
                if pv_pos > pv_all * 1.2:
                    add("Enhance care coordination and follow-up for high-utilizers (frequent prior visits). Set up case management and social support referrals.")

            # Length of stay
            if "length_of_stay" in df_eng.columns and pos_mask.any():
                los_pos = float(df_eng.loc[pos_mask, "length_of_stay"].mean())
                los_all = float(df_eng["length_of_stay"].mean())
                if los_pos > los_all * 1.1:
                    add("Strengthen discharge planning for longer stays: clear instructions, early follow-up appointment, and home health checks where appropriate.")

            # Medication burden
            if "medication_count" in df_eng.columns and pos_mask.any():
                med_pos = float(df_eng.loc[pos_mask, "medication_count"].mean())
                med_all = float(df_eng["medication_count"].mean())
                if med_pos > med_all * 1.2:
                    add("Perform medication reconciliation and adherence counseling. Simplify regimens and check for affordability.")

            # Emergency visits
            if "number_emergency" in df_eng.columns and pos_mask.any():
                er_pos = float(df_eng.loc[pos_mask, "number_emergency"].mean())
                er_all = float(df_eng["number_emergency"].mean())
                if er_pos > er_all * 1.2:
                    add("Create an urgent-access clinic pathway to divert avoidable ED use and provide rapid outpatient escalation.")

            if suggestions:
                for s in suggestions:
                    st.write(f"- {s}")
            else:
                st.write("- Maintain standard discharge best practices: timely follow-up, clear instructions, and patient education.")
        except Exception as e:
            st.info(f"Suggestions not available due to: {e}")

        status.update(
            label="Analysis complete. All relevant information displayed.",
            state="complete",
        )

st.sidebar.markdown("---")
st.sidebar.write("Health: ✅ App ready")
