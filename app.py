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
import requests

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Patient No-Show Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

API_URL = "http://127.0.0.1:8000"

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

def check_api():
    """Check if FastAPI is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────

try:
    scored_df = load_predictions()
    top30_df  = load_top30()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏥 Patient No-Show Risk Dashboard</h1>
    <p>Identify high-risk patients and prioritize proactive outreach before their appointment.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab1, tab2 = st.tabs(["📊 Dashboard", "🔬 Live Predict"])


# ══════════════════════════════════════════════
# TAB 1 — Dashboard (existing)
# ══════════════════════════════════════════════

with tab1:

    if not data_loaded:
        st.error("Required files not found. Please run `model_training.py` and `predict.py` first.")
    else:

        # ── KPI Cards ───────────────────────────────

        total        = len(scored_df)
        high_risk    = int((scored_df["noshow_probability"] > 70).sum())
        predicted_ns = int((scored_df["noshow_probability"] > 50).sum())
        noshow_rate  = round(predicted_ns / total * 100, 1)

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
                        prob                     = row["noshow_probability"]
                        risk_label, risk_cls     = get_risk_level(prob)
                        action_label, action_cls = get_action(prob)
                        appt_day                 = str(row.get("appointment_day", "N/A"))[:10]
                        risk_factors             = row.get("shap_risk_factors", "N/A")

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

        risk_filter = st.radio(
            "Filter by risk level",
            options=["All", "High (>70%)", "Medium (50–70%)", "Low (<50%)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        display_df = top30_df.copy()
        display_df.columns = [
            "Patient ID", "Appointment ID", "Appointment Day",
            "No-Show Prob (%)", "Top Risk Factors"
        ]
        display_df["Patient ID"]     = display_df["Patient ID"].apply(lambda x: f"{x:.0f}")
        display_df["Appointment ID"] = display_df["Appointment ID"].apply(lambda x: f"{x:.0f}")
        display_df["Risk Level"]     = display_df["No-Show Prob (%)"].apply(lambda p: get_risk_level(p)[0])
        display_df["Action"]         = display_df["No-Show Prob (%)"].apply(lambda p: get_action(p)[0])

        if risk_filter == "High (>70%)":
            display_df = display_df[display_df["No-Show Prob (%)"] > 70]
        elif risk_filter == "Medium (50–70%)":
            display_df = display_df[(display_df["No-Show Prob (%)"] > 50) & (display_df["No-Show Prob (%)"] <= 70)]
        elif risk_filter == "Low (<50%)":
            display_df = display_df[display_df["No-Show Prob (%)"] <= 50]

        styled = display_df.style.applymap(color_prob, subset=["No-Show Prob (%)"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
        st.caption(f"Showing {len(display_df)} patient(s)")

        st.markdown("---")
        st.caption(
            "Patient No-Show Prediction System | ML Design Final Project | "
            "University of Cincinnati, Lindner College of Business"
        )


# ══════════════════════════════════════════════
# TAB 2 — Live Predict (calls FastAPI)
# ══════════════════════════════════════════════

with tab2:

    st.markdown('<div class="section-header">🔬 Live Patient Prediction</div>', unsafe_allow_html=True)
    st.caption("Enter patient details to get a real-time no-show prediction from the model API.")

    # ── API status ──
    api_online = check_api()
    if api_online:
        st.success("✅ FastAPI is connected and running at http://127.0.0.1:8000")
    else:
        st.error("❌ FastAPI is not running. Start it with: `uvicorn api.main:app --reload --port 8000`")

    st.markdown("")

    # ── Input form ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Patient Info**")
        gender       = st.selectbox("Gender",                   [("Female", 1), ("Male", 0)],    format_func=lambda x: x[0])
        age          = st.number_input("Age",                   min_value=0, max_value=120, value=35)
        scholarship  = st.selectbox("Bolsa Família Scholarship", [("No", 0), ("Yes", 1)],         format_func=lambda x: x[0])

    with col2:
        st.markdown("**Health Conditions**")
        hipertension = st.selectbox("Hypertension", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        diabetes     = st.selectbox("Diabetes",     [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        alcoholism   = st.selectbox("Alcoholism",   [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        handcap      = st.selectbox("Disability",   [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

    with col3:
        st.markdown("**Appointment Details**")
        sms_received = st.selectbox("SMS Reminder Sent", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        lead_time    = st.number_input("Days until appointment", min_value=0, max_value=365, value=7)
        day_of_week  = st.selectbox("Appointment Day",
                           [(d, i) for i, d in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])],
                           format_func=lambda x: x[0])
        hour_of_day  = st.number_input("Hour scheduled (0–23)", min_value=0, max_value=23, value=10)
        neighbourhood_noshow_rate = st.slider("Neighbourhood no-show rate", 0.0, 1.0, 0.20, 0.01)

    is_weekend = 1 if day_of_week[1] >= 5 else 0

    st.markdown("")

    # ── Predict button ──
    if st.button("🔍 Predict No-Show Risk", type="primary", disabled=not api_online):

        payload = {
            "gender":                    gender[1],
            "age":                       int(age),
            "scholarship":               scholarship[1],
            "hipertension":              hipertension[1],
            "diabetes":                  diabetes[1],
            "alcoholism":                alcoholism[1],
            "handcap":                   handcap[1],
            "sms_received":              sms_received[1],
            "date_diff":                 int(lead_time),
            "lead_time":                 int(lead_time),
            "day_of_week":               day_of_week[1],
            "is_weekend":                is_weekend,
            "hour_of_day":               int(hour_of_day),
            "neighbourhood_noshow_rate": float(neighbourhood_noshow_rate),
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
                prob   = result["noshow_probability"]
                risk   = result["risk_level"]
                action = result["recommended_action"]
                pred   = result["prediction"]

                st.markdown("---")
                st.markdown("### Prediction Result")

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("No-Show Probability", f"{prob}%")
                with r2:
                    st.metric("Risk Level", risk)
                with r3:
                    st.metric("Prediction", pred.title())
                with r4:
                    st.metric("Recommended Action", action)

                st.progress(min(prob / 100, 1.0))

                if risk == "High":
                    st.error("⚠️ High risk patient — recommend calling immediately.")
                elif risk == "Medium":
                    st.warning("📱 Medium risk — send an SMS reminder.")
                else:
                    st.success("✅ Low risk — monitor only.")

            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Make sure it is running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    st.markdown("---")
    st.caption(
        "Patient No-Show Prediction System | ML Design Final Project | "
        "University of Cincinnati, Lindner College of Business"
    )
