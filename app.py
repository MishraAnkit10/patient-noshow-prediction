"""
Streamlit Dashboard: Patient No-Show Risk Dashboard
Displays no-show risk scores and patient lookup for clinical staff.

Author: Ankit Mishra, Kate Hattemer, Claude
Run: streamlit run app.py
"""

import os
import pickle
import pandas as pd
import streamlit as st

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
        background: #1B3A5C;
        padding: 1.25rem 1.75rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 600;
    }
    .main-header p {
        color: #B0C4DE;
        margin: 0.25rem 0 0;
        font-size: 0.85rem;
    }
    .kpi-card {
        background: #F8F9FA;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        text-align: center;
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 600;
    }
    .kpi-card p {
        color: #6B7280;
        font-size: 0.8rem;
        margin: 0.2rem 0 0;
    }
    .kpi-high { color: #B91C1C; }
    .kpi-blue { color: #1B3A5C; }
    .kpi-amber { color: #B45309; }
    .badge-high {
        background: #FEE2E2;
        color: #B91C1C;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-med {
        background: #FEF3C7;
        color: #B45309;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low {
        background: #DCFCE7;
        color: #166534;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .section-header {
        color: #1B3A5C;
        border-bottom: 2px solid #2E6B9E;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-box {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background: white;
    }
    .action-call {
        background: #FEE2E2;
        color: #B91C1C;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .action-sms {
        background: #FEF3C7;
        color: #B45309;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .action-monitor {
        background: #DCFCE7;
        color: #166534;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

@st.cache_data
def load_predictions():
    return pd.read_csv("predictions/scored_appointments.csv", index_col=0)

@st.cache_data
def load_top30():
    return pd.read_csv("predictions/top_30_high_risk.csv", index_col=0)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def get_risk_level(prob):
    if prob > 70:
        return "High", "badge-high"
    elif prob > 50:
        return "Medium", "badge-med"
    else:
        return "Low", "badge-low"

def get_action(prob):
    if prob > 70:
        return "Call now", "action-call"
    elif prob > 50:
        return "Send SMS", "action-sms"
    else:
        return "Monitor", "action-monitor"

def color_prob(val):
    if val > 70:
        return "background-color: #FEE2E2; color: #B91C1C; font-weight: bold"
    elif val > 50:
        return "background-color: #FEF3C7; color: #B45309; font-weight: bold"
    return "background-color: #DCFCE7; color: #166534; font-weight: bold"


# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────

try:
    scored_df = load_predictions()
    top30_df  = load_top30()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(
        f"Required files not found. Please run `model_training.py` and `predict.py` first.\n\nMissing: {e}"
    )


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────

if data_loaded:

    # ── Header ──────────────────────────────────

    st.markdown("""
    <div class="main-header">
        <h1>🏥 Patient No-Show Risk Dashboard</h1>
        <p>Identify high-risk patients and prioritize proactive outreach before their appointment.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ───────────────────────────────

    total          = len(scored_df)
    high_risk      = int((scored_df["noshow_probability"] > 70).sum())
    predicted_ns   = int((scored_df["noshow_probability"] > 50).sum())
    noshow_rate    = round(predicted_ns / total * 100, 1)

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3 class="kpi-blue">{total:,}</h3>
            <p>Total appointments</p>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <h3 class="kpi-high">{high_risk:,}</h3>
            <p>High risk (&gt;70% probability)</p>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <h3 class="kpi-amber">{noshow_rate}%</h3>
            <p>Predicted no-show rate (&gt;50%)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Patient Lookup ───────────────────────────

    st.markdown('<div class="section-header">Patient Lookup</div>', unsafe_allow_html=True)
    st.caption("Enter a Patient ID to view their risk score, appointment details, and recommended action.")

    patient_input = st.text_input("Patient ID", placeholder="e.g. 38786636279984", label_visibility="collapsed")

    if patient_input:
        try:
            pid = float(patient_input)
            matches = scored_df[scored_df["patient_id"] == pid]

            if len(matches) == 0:
                st.warning(f"No appointments found for Patient ID: {patient_input}")
            else:
                st.success(f"Found {len(matches)} appointment(s) for Patient ID: {patient_input}")

                for _, row in matches.iterrows():
                    prob              = row["noshow_probability"]
                    risk_label, risk_cls  = get_risk_level(prob)
                    action_label, action_cls = get_action(prob)
                    appt_day          = str(row.get("appointment_day", "N/A"))[:10]
                    risk_factors      = row.get("shap_risk_factors", "N/A")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("No-Show Probability", f"{prob:.1f}%")
                    with c2:
                        st.metric("Risk Level", risk_label)
                    with c3:
                        st.metric("Appointment Day", appt_day)
                    with c4:
                        st.metric("Recommended Action", action_label)

                    st.progress(min(prob / 100, 1.0))
                    st.caption(f"**Risk factors:** {risk_factors}")
                    st.divider()

        except ValueError:
            st.error("Please enter a valid numeric Patient ID.")

    # ── Top 30 High-Risk Table ───────────────────

    st.markdown('<div class="section-header">Top 30 High-Risk Patients</div>', unsafe_allow_html=True)
    st.caption("Patients most likely to miss their appointment — prioritize for proactive call or SMS outreach.")

    # Risk filter
    risk_filter = st.radio(
        "Filter by risk level",
        options=["All", "High (>70%)", "Medium (50–70%)", "Low (<50%)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    display_df = top30_df.copy()

    # Rename columns for display
    display_df.columns = [
        "Patient ID", "Appointment ID", "Appointment Day",
        "No-Show Prob (%)", "Top Risk Factors"
    ]

    # Format IDs as integers
    display_df["Patient ID"]      = display_df["Patient ID"].apply(lambda x: f"{x:.0f}")
    display_df["Appointment ID"]  = display_df["Appointment ID"].apply(lambda x: f"{x:.0f}")

    # Add risk level column
    display_df["Risk Level"] = display_df["No-Show Prob (%)"].apply(
        lambda p: get_risk_level(p)[0]
    )

    # Add recommended action column
    display_df["Action"] = display_df["No-Show Prob (%)"].apply(
        lambda p: get_action(p)[0]
    )

    # Apply filter
    if risk_filter == "High (>70%)":
        display_df = display_df[display_df["No-Show Prob (%)"] > 70]
    elif risk_filter == "Medium (50–70%)":
        display_df = display_df[
            (display_df["No-Show Prob (%)"] > 50) & (display_df["No-Show Prob (%)"] <= 70)
        ]
    elif risk_filter == "Low (<50%)":
        display_df = display_df[display_df["No-Show Prob (%)"] <= 50]

    # Style the probability column
    styled = (
        display_df.style
        .applymap(color_prob, subset=["No-Show Prob (%)"])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
    st.caption(f"Showing {len(display_df)} patient(s)")

    # ── Footer ───────────────────────────────────

    st.markdown("---")
    st.caption(
        "Patient No-Show Prediction System | ML Design Final Project | "
        "University of Cincinnati, Lindner College of Business"
    )