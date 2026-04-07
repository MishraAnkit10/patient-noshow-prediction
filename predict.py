"""
Prediction: Patient No-Show Risk Scoring
Scores all appointments and generates ranked high-risk patient lists.

Author: Ankit Mishra
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("predict.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

GOLD_PATH = "gold/ml_ready_appointments.parquet"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
METADATA_COLS = ["patient_id", "appointment_id", "appointment_day"]
TARGET_COL = "showed_up"
TOP_N = 30


def load_artifacts():
    """Load saved model and metadata."""
    logger.info("Loading model artifacts...")

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    logger.info(f"Best model: {metadata['best_model_name']}")
    logger.info(f"Trained at: {metadata['trained_at']}")
    return model, metadata


def compute_shap_reasons(model, X, feature_names, top_k=3):
    """Compute per-patient top SHAP risk factors for no-show."""
    logger.info("Computing SHAP values for all predictions...")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_processed = preprocessor.transform(X)

    # Get feature names after preprocessing
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out())
    num_features = list(preprocessor.named_transformers_["num"].get_feature_names_out())
    all_feature_names = num_features + cat_feature_names

    # SHAP explainer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, X_processed)
    else:
        explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_processed)

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]  # No-show class
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 0]  # No-show class
    else:
        shap_vals = shap_values

    # For each patient, get top-k features contributing to no-show risk
    reasons_list = []
    for i in range(len(X)):
        patient_shap = shap_vals[i]
        # Sort by absolute value (highest impact on no-show)
        top_indices = np.argsort(np.abs(patient_shap))[::-1][:top_k]
        reasons = []
        for idx in top_indices:
            fname = all_feature_names[idx]
            val = patient_shap[idx]
            reasons.append(f"{fname}: {val:+.4f}")
        reasons_list.append(", ".join(reasons))

    logger.info(f"SHAP reasons computed for {len(X):,} patients")
    return reasons_list


def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   PREDICTION: Patient No-Show Risk Scoring               ║")
    logger.info("╚" + "═" * 58 + "╝")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Load model
    model, model_meta = load_artifacts()

    # Load Gold data
    df = pd.read_parquet(GOLD_PATH)
    logger.info(f"Loaded Gold data: {df.shape[0]:,} rows")

    # Separate metadata and features
    meta = df[METADATA_COLS].copy()
    feature_cols = [c for c in df.columns if c not in METADATA_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y_actual = df[TARGET_COL]

    # Score all appointments
    logger.info("Scoring all appointments...")
    y_prob = model.predict_proba(X)

    # prob[:, 0] = probability of no-show, prob[:, 1] = probability of showing up
    noshow_prob = y_prob[:, 0]

    # Compute SHAP reasons
    shap_reasons = compute_shap_reasons(model, X, feature_cols)

    # Build scored dataset
    scored = meta.copy()
    scored["actual_showed_up"] = y_actual.values
    scored["noshow_probability"] = np.round(noshow_prob * 100, 2)
    scored["noshow_probability_raw"] = noshow_prob
    scored["shap_risk_factors"] = shap_reasons

    # Sort by no-show probability descending
    scored = scored.sort_values("noshow_probability", ascending=False).reset_index(drop=True)
    scored.index = scored.index + 1  # 1-based ranking
    scored.index.name = "rank"

    # Save full scored dataset
    full_path = os.path.join(PREDICTIONS_DIR, "scored_appointments.csv")
    scored.to_csv(full_path)
    logger.info(f"Full scored dataset saved: {full_path} ({len(scored):,} rows)")

    # Top 30 high-risk patients
    top30 = scored.head(TOP_N).copy()
    top30_clean = top30[["patient_id", "appointment_id", "appointment_day",
                          "noshow_probability", "shap_risk_factors"]].copy()
    top30_clean.columns = ["Patient ID", "Appointment ID", "Appointment Day",
                            "No-Show Probability (%)", "Top 3 SHAP Risk Factors"]

    top30_path = os.path.join(PREDICTIONS_DIR, "top_30_high_risk.csv")
    top30_clean.to_csv(top30_path)
    logger.info(f"Top {TOP_N} high-risk patients saved: {top30_path}")

    # Log top 10 for quick view
    logger.info("-" * 60)
    logger.info(f"TOP 10 HIGH-RISK PATIENTS:")
    for idx, row in top30_clean.head(10).iterrows():
        logger.info(f"  Rank {idx}: Patient {row['Patient ID']:.0f} | "
                    f"P(no-show)={row['No-Show Probability (%)']:.1f}% | "
                    f"{row['Top 3 SHAP Risk Factors']}")

    # Summary stats
    logger.info("-" * 60)
    logger.info("PREDICTION SUMMARY:")
    threshold = 50  # %
    predicted_noshow = (scored["noshow_probability"] > threshold).sum()
    logger.info(f"  Total appointments scored: {len(scored):,}")
    logger.info(f"  Predicted no-shows (>{threshold}%): {predicted_noshow:,}")
    logger.info(f"  Predicted no-show rate: {predicted_noshow / len(scored) * 100:.1f}%")
    logger.info(f"  Mean no-show probability: {scored['noshow_probability'].mean():.1f}%")
    logger.info(f"  Median no-show probability: {scored['noshow_probability'].median():.1f}%")
    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE")


if __name__ == "__main__":
    main()
