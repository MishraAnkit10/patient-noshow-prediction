"""
Model Training: Patient No-Show Prediction
Trains Logistic Regression and Decision Tree baselines with MLflow tracking.

Author: Ankit Mishra
"""

import os
import time
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve,
)

import shap
import mlflow
import mlflow.sklearn

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
        logging.FileHandler("model_training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

GOLD_PATH = "gold/ml_ready_appointments.parquet"
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

METADATA_COLS = ["patient_id", "appointment_id", "appointment_day"]
TARGET_COL = "showed_up"

# For no-show prediction, we invert: predict no-show (0) as the positive class
# showed_up=1 means showed, showed_up=0 means no-show
# We want recall on the NO-SHOW class (label 0)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ══════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ══════════════════════════════════════════════

def load_and_prepare_data():
    """Load Gold parquet and separate features, target, metadata."""
    logger.info("=" * 60)
    logger.info("DATA LOADING & PREPARATION")
    logger.info("=" * 60)

    df = pd.read_parquet(GOLD_PATH)
    logger.info(f"Loaded Gold data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Separate metadata
    metadata = df[METADATA_COLS].copy()
    logger.info(f"Metadata columns separated: {METADATA_COLS}")

    # Separate target
    y = df[TARGET_COL].copy()
    logger.info(f"Target: {TARGET_COL}")
    logger.info(f"  Class distribution: showed_up=1: {(y == 1).sum():,} ({(y == 1).mean() * 100:.1f}%), "
                f"no-show=0: {(y == 0).sum():,} ({(y == 0).mean() * 100:.1f}%)")

    # Separate features (exclude metadata + target)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS + [TARGET_COL]]
    X = df[feature_cols].copy()
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Identify column types
    categorical_cols = ["age_group"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

    return X, y, metadata, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Build sklearn ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ══════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════

def get_models(preprocessor):
    """Return dict of model name -> sklearn Pipeline."""
    models = {
        "logreg_balanced_v1": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
                C=1.0,
            )),
        ]),
        "dtree_depth5_v1": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=5,
                random_state=RANDOM_STATE,
                min_samples_split=20,
                min_samples_leaf=10,
            )),
        ]),
    }
    return models


# ══════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════

def evaluate_model(model, X, y, model_name):
    """Evaluate model using stratified k-fold CV. Returns metrics dict."""
    logger.info(f"Evaluating {model_name} with {CV_FOLDS}-fold stratified CV...")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Cross-validated predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=skf, method="predict")
    y_prob_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    # Metrics — we focus on NO-SHOW class (label 0)
    # For recall on no-show: pos_label=0
    metrics = {
        "auc_roc": roc_auc_score(y, y_prob_cv),
        "f1_noshow": f1_score(y, y_pred_cv, pos_label=0),
        "precision_noshow": precision_score(y, y_pred_cv, pos_label=0),
        "recall_noshow": recall_score(y, y_pred_cv, pos_label=0),
        "f1_weighted": f1_score(y, y_pred_cv, average="weighted"),
    }

    cm = confusion_matrix(y, y_pred_cv)

    logger.info(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    logger.info(f"  Recall (no-show):  {metrics['recall_noshow']:.4f}")
    logger.info(f"  Precision (no-show): {metrics['precision_noshow']:.4f}")
    logger.info(f"  F1 (no-show):      {metrics['f1_noshow']:.4f}")
    logger.info(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")

    return metrics, cm, y_prob_cv


def plot_confusion_matrix(cm, model_name, save_path):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No-Show (0)", "Showed (1)"])
    ax.set_yticklabels(["No-Show (0)", "Showed (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=14)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved: {save_path}")


def plot_roc_curves(results, y, save_path):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_prob_cv"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['auc_roc']:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve comparison saved: {save_path}")


# ══════════════════════════════════════════════
# SHAP ANALYSIS
# ══════════════════════════════════════════════

def compute_shap(model, X_train, X_test, feature_names, model_name, save_path):
    """Compute and save SHAP summary plot."""
    logger.info(f"  Computing SHAP values for {model_name}...")

    # Get preprocessed data
    preprocessor = model.named_steps["preprocessor"]
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out())
    num_features = list(preprocessor.named_transformers_["num"].get_feature_names_out())
    all_feature_names = num_features + cat_feature_names

    classifier = model.named_steps["classifier"]

    # Use appropriate SHAP explainer
    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, X_train_processed)
    else:
        explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_test_processed)

    # For binary classification, shap_values might be a list [class0, class1]
    # or a 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]  # No-show class
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 0]  # No-show class
    else:
        shap_vals = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_test_processed,
        feature_names=all_feature_names,
        show=False, max_display=15,
    )
    plt.title(f"SHAP Summary: {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  SHAP summary plot saved: {save_path}")

    # Feature importance (mean absolute SHAP)
    importance = pd.DataFrame({
        "feature": all_feature_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info(f"  Top 10 features by SHAP importance:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

    return shap_vals, all_feature_names, importance


# ══════════════════════════════════════════════
# MLFLOW TRACKING
# ══════════════════════════════════════════════

def log_to_mlflow(model_name, model, metrics, cm_path, shap_path, importance_df, params, train_time):
    """Log a model run to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("training_time_seconds", train_time)

        # Log artifacts
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path)
        if os.path.exists(shap_path):
            mlflow.log_artifact(shap_path)

        # Log feature importance as CSV
        imp_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_feature_importance.csv")
        importance_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(imp_path)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"  MLflow run logged: {model_name}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   MODEL TRAINING: Patient No-Show Prediction             ║")
    logger.info("║   Logistic Regression + Decision Tree Baselines          ║")
    logger.info("╚" + "═" * 58 + "╝")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Load data ──
    X, y, metadata, numeric_cols, categorical_cols = load_and_prepare_data()

    # ── Build preprocessor ──
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info(f"Train/test split: {len(X_train):,} train / {len(X_test):,} test (stratified)")

    # ── Get models ──
    models = get_models(preprocessor)

    # ── MLflow setup ──
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("patient-noshow-prediction")

    # ── Train and evaluate each model ──
    results = {}

    for model_name, model in models.items():
        logger.info("=" * 60)
        logger.info(f"TRAINING: {model_name}")
        logger.info("=" * 60)

        # Train
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        logger.info(f"  Training time: {train_time:.2f}s")

        # Evaluate via CV on FULL data (standard practice for comparison)
        metrics, cm, y_prob_cv = evaluate_model(model, X, y, model_name)

        # Confusion matrix plot
        cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, model_name, cm_path)

        # SHAP
        shap_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_shap_summary.png")
        shap_vals, feature_names, importance_df = compute_shap(
            model, X_train, X_test, numeric_cols + categorical_cols, model_name, shap_path,
        )

        # Get hyperparameters for logging
        classifier = model.named_steps["classifier"]
        params = classifier.get_params()

        # Log to MLflow
        log_to_mlflow(model_name, model, metrics, cm_path, shap_path, importance_df, params, train_time)

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "cm": cm,
            "y_prob_cv": y_prob_cv,
            "importance": importance_df,
            "shap_vals": shap_vals,
            "feature_names": feature_names,
            "train_time": train_time,
        }

    # ── ROC curve comparison ──
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve_comparison.png")
    plot_roc_curves(results, y, roc_path)

    # ── Select best model (by recall on no-show class) ──
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    comparison = []
    for name, res in results.items():
        m = res["metrics"]
        comparison.append({
            "model": name,
            "auc_roc": m["auc_roc"],
            "recall_noshow": m["recall_noshow"],
            "precision_noshow": m["precision_noshow"],
            "f1_noshow": m["f1_noshow"],
            "train_time": res["train_time"],
        })
        logger.info(f"  {name}: AUC={m['auc_roc']:.4f}, Recall={m['recall_noshow']:.4f}, "
                    f"Precision={m['precision_noshow']:.4f}, F1={m['f1_noshow']:.4f}")

    # Save comparison table
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index=False)

    best_name = max(results, key=lambda n: results[n]["metrics"]["recall_noshow"])
    best_model = results[best_name]["model"]
    logger.info(f"\n  BEST MODEL (by recall): {best_name}")
    logger.info(f"  Recall (no-show): {results[best_name]['metrics']['recall_noshow']:.4f}")

    # ── Save best model artifacts ──
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model saved: {MODELS_DIR}/best_model.pkl")

    # Save preprocessor separately (it's inside the pipeline, but useful for reference)
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(fitted_preprocessor, f)
    logger.info(f"Preprocessor saved: {MODELS_DIR}/preprocessor.pkl")

    # Save feature names and model metadata
    model_meta = {
        "best_model_name": best_name,
        "metrics": results[best_name]["metrics"],
        "feature_names": results[best_name]["feature_names"],
        "feature_columns": list(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump(model_meta, f)
    logger.info(f"Model metadata saved: {MODELS_DIR}/model_metadata.pkl")

    # Save all results for Streamlit
    with open(os.path.join(MODELS_DIR, "all_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
