"""
Streamlit Dashboard: Patient No-Show Risk Dashboard
Displays predictions, model performance, and patient lookup.

Author: Ankit Mishra
Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Patient No-Show Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E6B9E 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #B0C4DE; margin: 0.3rem 0 0; font-size: 0.9rem; }
    .kpi-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-card h3 { color: #1B3A5C; font-size: 1.8rem; margin: 0; }
    .kpi-card p { color: #666; font-size: 0.85rem; margin: 0.3rem 0 0; }
    .risk-high { background-color: #FFCCCC; color: #CC0000; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .risk-medium { background-color: #FFE0B2; color: #E65100; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .risk-low { background-color: #FFF9C4; color: #F57F17; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .section-header {
        color: #1B3A5C;
        border-bottom: 2px solid #2E6B9E;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data Loading (cached)
# ──────────────────────────────────────────────

@st.cache_data
def load_predictions():
    return pd.read_csv("predictions/scored_appointments.csv", index_col=0)

@st.cache_data
def load_top30():
    return pd.read_csv("predictions/top_30_high_risk.csv", index_col=0)

@st.cache_resource
def load_model_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/model_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    with open("models/all_results.pkl", "rb") as f:
        results = pickle.load(f)
    return model, meta, results


# ──────────────────────────────────────────────
# Load all data
# ──────────────────────────────────────────────

try:
    scored_df = load_predictions()
    top30_df = load_top30()
    model, model_meta, all_results = load_model_artifacts()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"Required files not found. Please run `model_training.py` and `predict.py` first.\n\nMissing: {e}")


if data_loaded:

    # ══════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════

    st.markdown(f"""
    <div class="main-header">
        <h1>Patient No-Show Risk Dashboard</h1>
        <p>Model: {model_meta['best_model_name']} | Trained: {model_meta['trained_at'][:19]} |
        Features: {len(model_meta['feature_columns'])}</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # KPI CARDS
    # ══════════════════════════════════════════

    total = len(scored_df)
    threshold = 50
    predicted_noshow = (scored_df["noshow_probability"] > threshold).sum()
    noshow_rate = predicted_noshow / total * 100
    auc_score = model_meta["metrics"]["auc_roc"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><h3>{total:,}</h3><p>Total Appointments</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><h3>{predicted_noshow:,}</h3><p>Predicted No-Shows (>{threshold}%)</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><h3>{noshow_rate:.1f}%</h3><p>Predicted No-Show Rate</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><h3>{auc_score:.4f}</h3><p>Model AUC-ROC</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════
    # TOP 30 HIGH-RISK PATIENTS
    # ══════════════════════════════════════════

    st.markdown('<h2 class="section-header">Top 30 High-Risk Patients</h2>', unsafe_allow_html=True)
    st.caption("Patients most likely to miss their appointment — prioritize for proactive call/SMS outreach.")

    # Build display dataframe
    display_df = top30_df.copy()
    display_df.columns = ["Patient ID", "Appointment ID", "Appointment Day",
                          "No-Show Prob (%)", "Top 3 SHAP Risk Factors"]

    # Format patient ID as integer string
    display_df["Patient ID"] = display_df["Patient ID"].apply(lambda x: f"{x:.0f}")
    display_df["Appointment ID"] = display_df["Appointment ID"].apply(lambda x: f"{x:.0f}")

    def color_risk(val):
        if val > 70:
            return "background-color: #FFCCCC; color: #CC0000; font-weight: bold"
        elif val > 50:
            return "background-color: #FFE0B2; color: #E65100; font-weight: bold"
        elif val > 30:
            return "background-color: #FFF9C4; color: #F57F17; font-weight: bold"
        return ""

    styled = display_df.style.applymap(color_risk, subset=["No-Show Prob (%)"])
    st.dataframe(styled, use_container_width=True, height=600)

    # ══════════════════════════════════════════
    # MODEL PERFORMANCE COMPARISON
    # ══════════════════════════════════════════

    st.markdown('<h2 class="section-header">Model Performance Comparison</h2>', unsafe_allow_html=True)

    col_metrics, col_roc = st.columns([1, 1.2])

    with col_metrics:
        st.subheader("Metrics Comparison")
        comp_data = []
        for name, res in all_results.items():
            m = res["metrics"]
            comp_data.append({
                "Model": name,
                "AUC-ROC": f"{m['auc_roc']:.4f}",
                "Recall (No-Show)": f"{m['recall_noshow']:.4f}",
                "Precision (No-Show)": f"{m['precision_noshow']:.4f}",
                "F1 (No-Show)": f"{m['f1_noshow']:.4f}",
                "F1 (Weighted)": f"{m['f1_weighted']:.4f}",
                "Train Time (s)": f"{res['train_time']:.2f}",
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Highlight best
        best_name = model_meta["best_model_name"]
        st.success(f"**Best model (by recall):** {best_name} — Recall = {all_results[best_name]['metrics']['recall_noshow']:.4f}")

    with col_roc:
        st.subheader("ROC Curves")
        if os.path.exists("artifacts/roc_curve_comparison.png"):
            st.image("artifacts/roc_curve_comparison.png", use_container_width=True)
        else:
            st.info("ROC curve plot not found. Run model_training.py to generate.")

    # Confusion matrices side by side
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(all_results))
    for i, (name, res) in enumerate(all_results.items()):
        with cm_cols[i]:
            cm_path = f"artifacts/{name}_confusion_matrix.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption=name, use_container_width=True)

    # ══════════════════════════════════════════
    # FEATURE IMPORTANCE (SHAP)
    # ══════════════════════════════════════════

    st.markdown('<h2 class="section-header">Feature Importance (SHAP)</h2>', unsafe_allow_html=True)

    shap_cols = st.columns(len(all_results))
    for i, (name, res) in enumerate(all_results.items()):
        with shap_cols[i]:
            shap_path = f"artifacts/{name}_shap_summary.png"
            if os.path.exists(shap_path):
                st.image(shap_path, caption=f"SHAP Summary: {name}", use_container_width=True)

    # Top features bar chart for best model
    st.subheader(f"Top 10 Features — {best_name}")
    imp_df = all_results[best_name]["importance"].head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(imp_df["feature"][::-1], imp_df["mean_abs_shap"][::-1], color="#2E6B9E")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top 10 Features Driving No-Show Risk ({best_name})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ══════════════════════════════════════════
    # PATIENT LOOKUP
    # ══════════════════════════════════════════

    st.markdown('<h2 class="section-header">Patient Lookup</h2>', unsafe_allow_html=True)
    st.caption("Enter a Patient ID to view their individual risk score, appointment details, and risk factors.")

    patient_input = st.text_input("Patient ID", placeholder="e.g., 38786636279984")

    if patient_input:
        try:
            pid = float(patient_input)
            patient_rows = scored_df[scored_df["patient_id"] == pid]

            if len(patient_rows) == 0:
                st.warning(f"No appointments found for Patient ID: {patient_input}")
            else:
                st.success(f"Found {len(patient_rows)} appointment(s) for Patient ID: {patient_input}")

                for _, row in patient_rows.iterrows():
                    prob = row["noshow_probability"]
                    if prob > 70:
                        risk_label = "HIGH RISK"
                        risk_color = "#CC0000"
                    elif prob > 50:
                        risk_label = "MEDIUM RISK"
                        risk_color = "#E65100"
                    elif prob > 30:
                        risk_label = "LOW-MEDIUM RISK"
                        risk_color = "#F57F17"
                    else:
                        risk_label = "LOW RISK"
                        risk_color = "#2E7D32"

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("No-Show Probability", f"{prob:.1f}%")
                    with c2:
                        st.metric("Risk Level", risk_label)
                    with c3:
                        st.metric("Appointment Day", str(row.get("appointment_day", "N/A"))[:10])

                    # Risk factors
                    st.markdown(f"**SHAP Risk Factors:** {row.get('shap_risk_factors', 'N/A')}")

                    # Progress bar for probability
                    st.progress(min(prob / 100, 1.0))
                    st.divider()

        except ValueError:
            st.error("Please enter a valid numeric Patient ID.")

    # ══════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════

    st.markdown("---")
    st.caption("Patient No-Show Prediction System | ML Design Final Project | University of Cincinnati, Lindner College of Business")
