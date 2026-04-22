"""
FastAPI Backend: Patient No-Show Prediction API
Loads the trained model (MLflow or pickle fallback) and serves predictions.

Author: Gokul Reddy
Run: uvicorn api.main:app --reload --port 8000
"""

import os
import logging
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", "")  # Set via env var or .env

# Feature columns the model expects (same order as Gold parquet, excluding metadata + target)
FEATURE_COLUMNS = [
    "gender", "age", "scholarship", "hipertension", "diabetes",
    "alcoholism", "handcap", "sms_received", "date_diff", "lead_time",
    "day_of_week", "is_weekend", "hour_of_day", "age_group",
    "chronic_condition_count", "neighbourhood_noshow_rate",
]

AGE_BINS = [0, 18, 35, 55, 120]
AGE_LABELS = ["0-18", "19-35", "36-55", "56+"]

# ──────────────────────────────────────────────
# Global model holder
# ──────────────────────────────────────────────

model = None
model_metadata = None
model_source = None


def load_model():
    """Try loading model from MLflow first, then fall back to pickle."""
    global model, model_metadata, model_source

    # ── Attempt 1: MLflow ──
    if MLFLOW_RUN_ID:
        try:
            import mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.sklearn.load_model(f"runs:/{MLFLOW_RUN_ID}/model")
            model_source = f"mlflow (run: {MLFLOW_RUN_ID[:8]}...)"
            logger.info(f"Model loaded from MLflow run: {MLFLOW_RUN_ID}")
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}. Falling back to pickle.")
            model = None

    # ── Attempt 2: Pickle fallback ──
    if model is None:
        pkl_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"No model found. Either set MLFLOW_RUN_ID env var or ensure "
                f"{pkl_path} exists. Run model_training.py first."
            )
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        model_source = f"pickle ({pkl_path})"
        logger.info(f"Model loaded from pickle: {pkl_path}")

    # ── Load metadata (optional) ──
    meta_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            model_metadata = pickle.load(f)
        logger.info(f"Model metadata loaded: {model_metadata.get('best_model_name', 'unknown')}")


# ──────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    logger.info(f"API ready. Model source: {model_source}")
    yield
    logger.info("API shutting down.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Patient No-Show Prediction API",
    description=(
        "Predicts the probability that a patient will miss their appointment. "
        "Uses a trained sklearn pipeline (preprocessor + classifier) logged via MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit or any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────

class AppointmentInput(BaseModel):
    """Single appointment input for prediction."""
    gender: int = Field(..., ge=0, le=1, description="0 = Male, 1 = Female")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    scholarship: int = Field(..., ge=0, le=1, description="Enrolled in Bolsa Família (0/1)")
    hipertension: int = Field(..., ge=0, le=1, description="Has hypertension (0/1)")
    diabetes: int = Field(..., ge=0, le=1, description="Has diabetes (0/1)")
    alcoholism: int = Field(..., ge=0, le=1, description="Has alcoholism (0/1)")
    handcap: int = Field(..., ge=0, le=1, description="Has disability (0/1)")
    sms_received: int = Field(..., ge=0, le=1, description="Received SMS reminder (0/1)")
    date_diff: int = Field(..., ge=0, description="Raw date difference from original dataset (same as lead_time)")
    lead_time: int = Field(..., ge=0, description="Days between scheduling and appointment")
    day_of_week: int = Field(..., ge=0, le=6, description="Appointment day of week (0=Mon, 6=Sun)")
    is_weekend: int = Field(..., ge=0, le=1, description="Appointment on weekend (0/1)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour appointment was scheduled")
    neighbourhood_noshow_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Historical no-show rate for the patient's neighbourhood (0.0 to 1.0)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": 1,
                    "age": 42,
                    "scholarship": 0,
                    "hipertension": 1,
                    "diabetes": 0,
                    "alcoholism": 0,
                    "handcap": 0,
                    "sms_received": 1,
                    "date_diff": 15,
                    "lead_time": 15,
                    "day_of_week": 2,
                    "is_weekend": 0,
                    "hour_of_day": 10,
                    "neighbourhood_noshow_rate": 0.22,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Single prediction output."""
    noshow_probability: float = Field(..., description="Probability of no-show (0-100%)")
    risk_level: str = Field(..., description="High / Medium / Low")
    recommended_action: str = Field(..., description="Call now / Send SMS / Monitor")
    prediction: str = Field(..., description="no-show or will attend")


class BatchInput(BaseModel):
    """Batch of appointments for prediction."""
    appointments: list[AppointmentInput]


class BatchResponse(BaseModel):
    """Batch prediction output."""
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_source: str
    model_name: Optional[str]


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def _compute_derived_features(data: AppointmentInput) -> dict:
    """Compute age_group and chronic_condition_count from raw inputs."""
    # age_group
    age = data.age
    if age <= 18:
        age_group = "0-18"
    elif age <= 35:
        age_group = "19-35"
    elif age <= 55:
        age_group = "36-55"
    else:
        age_group = "56+"

    # chronic_condition_count
    chronic_count = data.hipertension + data.diabetes + data.alcoholism + data.handcap

    return {"age_group": age_group, "chronic_count": chronic_count}


def _build_dataframe(data: AppointmentInput) -> pd.DataFrame:
    """Convert a single AppointmentInput into a model-ready DataFrame."""
    derived = _compute_derived_features(data)
    row = {
        "gender": data.gender,
        "age": data.age,
        "scholarship": data.scholarship,
        "hipertension": data.hipertension,
        "diabetes": data.diabetes,
        "alcoholism": data.alcoholism,
        "handcap": data.handcap,
        "sms_received": data.sms_received,
        "date_diff": data.date_diff,
        "lead_time": data.lead_time,
        "day_of_week": data.day_of_week,
        "is_weekend": data.is_weekend,
        "hour_of_day": data.hour_of_day,
        "age_group": derived["age_group"],
        "chronic_condition_count": derived["chronic_count"],
        "neighbourhood_noshow_rate": data.neighbourhood_noshow_rate,
    }
    return pd.DataFrame([row])


def _classify(noshow_prob: float) -> dict:
    """Convert a no-show probability into risk level and action."""
    pct = round(noshow_prob * 100, 2)
    if pct > 70:
        risk, action = "High", "Call now"
    elif pct > 50:
        risk, action = "Medium", "Send SMS"
    else:
        risk, action = "Low", "Monitor"
    prediction = "no-show" if pct > 50 else "will attend"
    return {
        "noshow_probability": pct,
        "risk_level": risk,
        "recommended_action": action,
        "prediction": prediction,
    }


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"message": "Patient No-Show Prediction API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the model is loaded and ready."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="healthy",
        model_source=model_source or "unknown",
        model_name=model_metadata.get("best_model_name") if model_metadata else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(data: AppointmentInput):
    """
    Predict no-show probability for a single appointment.

    The model pipeline handles preprocessing internally (StandardScaler for
    numeric features, OneHotEncoder for age_group). Just pass raw feature values.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = _build_dataframe(data)
        proba = model.predict_proba(df)
        # proba[:, 0] = P(no-show), proba[:, 1] = P(showed up)
        noshow_prob = proba[0][0]
        return PredictionResponse(**_classify(noshow_prob))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(data: BatchInput):
    """
    Predict no-show probability for a batch of appointments.

    Useful for scoring an entire day's appointment list at once.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build a single DataFrame for all appointments (faster than looping)
        rows = []
        for appt in data.appointments:
            derived = _compute_derived_features(appt)
            rows.append({
                "gender": appt.gender,
                "age": appt.age,
                "scholarship": appt.scholarship,
                "hipertension": appt.hipertension,
                "diabetes": appt.diabetes,
                "alcoholism": appt.alcoholism,
                "handcap": appt.handcap,
                "sms_received": appt.sms_received,
                "date_diff": appt.date_diff,
                "lead_time": appt.lead_time,
                "day_of_week": appt.day_of_week,
                "is_weekend": appt.is_weekend,
                "hour_of_day": appt.hour_of_day,
                "age_group": derived["age_group"],
                "chronic_condition_count": derived["chronic_count"],
                "neighbourhood_noshow_rate": appt.neighbourhood_noshow_rate,
            })

        df = pd.DataFrame(rows)
        proba = model.predict_proba(df)
        noshow_probs = proba[:, 0]  # P(no-show) for each row

        predictions = [PredictionResponse(**_classify(p)) for p in noshow_probs]
        return BatchResponse(predictions=predictions, count=len(predictions))

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", tags=["Model Info"])
def model_info():
    """Return metadata about the loaded model."""
    if model_metadata is None:
        return {"message": "No metadata available. Model loaded from pickle without metadata."}
    return {
        "model_name": model_metadata.get("best_model_name"),
        "metrics": model_metadata.get("metrics"),
        "feature_columns": model_metadata.get("feature_columns"),
        "train_size": model_metadata.get("train_size"),
        "test_size": model_metadata.get("test_size"),
        "trained_at": model_metadata.get("trained_at"),
    }
