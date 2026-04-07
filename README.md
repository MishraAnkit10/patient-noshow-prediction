# Patient No-Show Prediction

ML pipeline for predicting healthcare appointment no-shows and generating actionable patient outreach lists.

**University of Cincinnati, Lindner College of Business — ML Design Final Project**

## Project Overview

This system uses a Medallion Architecture ETL pipeline (Bronze → Silver → Gold) to process ~107K healthcare appointment records, trains baseline classification models (Logistic Regression, Decision Tree) with MLflow experiment tracking, and produces a ranked list of high-risk patients for proactive call/SMS outreach.

## Project Structure

```
ML Design Final Project/
├── bronze/                          # Raw ingested data (parquet)
├── silver/                          # Cleaned + transformed data (parquet)
├── gold/                            # Feature-engineered ML-ready data (parquet)
├── models/                          # Saved model artifacts (pkl)
├── predictions/                     # Scored patients + top 30 high-risk list
├── artifacts/                       # SHAP plots, confusion matrices, ROC curves
├── mlruns/                          # MLflow experiment tracking data
├── Data_Ingestion.py                # ETL pipeline (Medallion Architecture)
├── model_training.py                # Model training + MLflow logging
├── predict.py                       # Score all patients + generate top 30 list
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MishraAnkit10/patient-noshow-prediction.git
cd patient-noshow-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the dataset (`healthcare_noshows_appt.csv`) in the project root.

## Running the Pipeline

Run scripts in order:

### Step 1: ETL Pipeline (Data Ingestion)
```bash
python Data_Ingestion.py
```
Produces `bronze/`, `silver/`, and `gold/` parquet files with validation checks at each layer.

### Step 2: Model Training
```bash
python model_training.py
```
Trains Logistic Regression and Decision Tree, logs to MLflow, saves best model to `models/`.

### Step 3: Generate Predictions
```bash
python predict.py
```
Scores all appointments and saves:
- `predictions/scored_appointments.csv` (full scored dataset)
- `predictions/top_30_high_risk.csv` (top 30 highest-risk patients with SHAP reasons)

### Step 4: Launch Dashboard
```bash
streamlit run app.py
```
Opens the interactive Patient No-Show Risk Dashboard in your browser.

### (Optional) View MLflow Dashboard
```bash
mlflow ui
```
Opens at `http://localhost:5000` to view experiment runs, metrics, and artifacts.

## Key Metrics

Models are evaluated on the **no-show class (label 0)** since the operational goal is catching patients who will miss their appointment:

- **Primary metric:** Recall (maximize no-show detection)
- **Secondary metric:** AUC-ROC (overall discrimination)
- **Success threshold:** AUC-ROC > 0.70, Recall > 0.65

## Dataset

Healthcare No-Shows Appointments Dataset from Kaggle — 107K appointments from April to June 2016 in Vitoria, Brazil.

**Features engineered:** lead_time, day_of_week, is_weekend, hour_of_day, age_group, chronic_condition_count, neighbourhood_noshow_rate

## Tech Stack

Python, Pandas, scikit-learn, SHAP, MLflow, Streamlit, PyArrow
