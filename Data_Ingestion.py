"""
ETL Pipeline: Patient No-Show Prediction
Medallion Architecture (Bronze → Silver → Gold)

Dataset: Healthcare No-Shows Appointments Dataset (~107K records)
Source: Kaggle

Author: Ankit Mishra
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etl_pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay",
    "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes",
    "Alcoholism", "Handcap", "SMS_received", "Showed_up", "Date.diff",
]

BOOLEAN_COLUMNS = [
    "Scholarship", "Hipertension", "Diabetes",
    "Alcoholism", "Handcap", "SMS_received", "Showed_up",
]

SNAKE_CASE_MAP = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hipertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handcap",
    "SMS_received": "sms_received",
    "Showed_up": "showed_up",
    "Date.diff": "date_diff",
}

AGE_BINS = [0, 18, 35, 55, 120]
AGE_LABELS = ["0-18", "19-35", "36-55", "56+"]

# Pipeline summary tracker
pipeline_summary = {
    "bronze_rows": 0,
    "silver_rows": 0,
    "gold_rows": 0,
    "bronze_validation": "NOT RUN",
    "silver_validation": "NOT RUN",
    "gold_validation": "NOT RUN",
    "gold_features": [],
}


# ══════════════════════════════════════════════
# BRONZE LAYER: Raw Ingestion
# ══════════════════════════════════════════════

def ingest_bronze(csv_path: str) -> pd.DataFrame:
    """
    Bronze Layer: Ingest raw CSV and save as parquet.
    No transformations applied — data is stored as-is.
    """
    logger.info("=" * 60)
    logger.info("BRONZE LAYER: Raw Data Ingestion")
    logger.info("=" * 60)

    # Read CSV
    logger.info(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Log raw stats
    logger.info(f"Raw row count: {len(df):,}")
    logger.info(f"Raw column count: {len(df.columns)}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    pipeline_summary["bronze_rows"] = len(df)

    # Save unmodified copy to bronze layer
    os.makedirs("bronze", exist_ok=True)
    bronze_path = "bronze/raw_appointments.parquet"
    df.to_parquet(bronze_path, index=False)
    logger.info(f"Bronze parquet saved: {bronze_path}")

    return df


# ──────────────────────────────────────────────
# BRONZE VALIDATION
# ──────────────────────────────────────────────

def validate_bronze(df: pd.DataFrame) -> bool:
    """
    Run validation checks on raw Bronze data.
    Returns True if all critical checks pass, False otherwise.
    """
    logger.info("-" * 60)
    logger.info("BRONZE VALIDATION: Running checks on raw data")
    logger.info("-" * 60)

    results = {}
    critical_fail = False

    # 1. Schema validation: confirm all 15 expected columns exist
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols:
        logger.error(f"SCHEMA CHECK — FAIL: Missing columns: {missing_cols}")
        results["schema"] = "FAIL"
        critical_fail = True
    else:
        logger.info(f"SCHEMA CHECK — PASS: All {len(EXPECTED_COLUMNS)} expected columns present")
        results["schema"] = "PASS"
    if extra_cols:
        logger.warning(f"SCHEMA CHECK — WARNING: Extra columns found: {extra_cols}")

    # 2. Data type validation
    dtype_issues = []

    if not pd.api.types.is_numeric_dtype(df["PatientId"]):
        dtype_issues.append("PatientId is not numeric")
    if not pd.api.types.is_integer_dtype(df["Age"]):
        dtype_issues.append("Age is not integer")

    gender_values = set(df["Gender"].dropna().unique())
    if not gender_values.issubset({"F", "M"}):
        dtype_issues.append(f"Gender has unexpected values: {gender_values - {'F', 'M'}}")

    for col in ["ScheduledDay", "AppointmentDay"]:
        try:
            pd.to_datetime(df[col].head(100))
        except Exception:
            dtype_issues.append(f"{col} is not parseable as datetime")

    for col in BOOLEAN_COLUMNS:
        if df[col].dtype != bool:
            dtype_issues.append(f"{col} is {df[col].dtype}, expected bool")

    if dtype_issues:
        for issue in dtype_issues:
            logger.error(f"DTYPE CHECK — FAIL: {issue}")
        results["dtype"] = "FAIL"
        critical_fail = True
    else:
        logger.info("DTYPE CHECK — PASS: All columns have expected data types")
        results["dtype"] = "PASS"

    # 3. Null check
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        for col, count in null_cols.items():
            logger.warning(f"NULL CHECK — WARNING: {col} has {count:,} null values")
        results["nulls"] = "WARNING"
    else:
        logger.info("NULL CHECK — PASS: No null values found in any column")
        results["nulls"] = "PASS"

    # 4. Duplicate check
    dup_count = df["AppointmentID"].duplicated().sum()
    if dup_count > 0:
        logger.warning(f"DUPLICATE CHECK — WARNING: {dup_count:,} duplicate AppointmentIDs found")
        results["duplicates"] = "WARNING"
    else:
        logger.info("DUPLICATE CHECK — PASS: No duplicate AppointmentIDs")
        results["duplicates"] = "PASS"

    # 5. Range checks
    range_issues = []

    invalid_age = ((df["Age"] < 0) | (df["Age"] > 120)).sum()
    if invalid_age > 0:
        range_issues.append(f"Age: {invalid_age:,} records outside 0-120 range")

    negative_datediff = (df["Date.diff"] < 0).sum()
    if negative_datediff > 0:
        range_issues.append(f"Date.diff: {negative_datediff:,} records with negative values")

    if range_issues:
        for issue in range_issues:
            logger.warning(f"RANGE CHECK — WARNING: {issue}")
        results["range"] = "WARNING"
    else:
        logger.info("RANGE CHECK — PASS: All values within expected ranges")
        results["range"] = "PASS"

    # Validation summary
    logger.info("-" * 40)
    logger.info("BRONZE VALIDATION SUMMARY:")
    for check, status in results.items():
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARNING" else "✗")
        logger.info(f"  {icon} {check}: {status}")

    if critical_fail:
        logger.error("BRONZE VALIDATION: CRITICAL CHECKS FAILED — Pipeline halted")
        pipeline_summary["bronze_validation"] = "FAIL"
        return False

    logger.info("BRONZE VALIDATION: ALL CRITICAL CHECKS PASSED")
    pipeline_summary["bronze_validation"] = "PASS"
    return True


# ══════════════════════════════════════════════
# SILVER LAYER: Cleaned + Transformed
# ══════════════════════════════════════════════

def transform_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silver Layer: Clean, transform, and standardize the data.
    """
    logger.info("=" * 60)
    logger.info("SILVER LAYER: Data Cleaning and Transformation")
    logger.info("=" * 60)

    initial_rows = len(df)
    rows_removed_log = []

    # 1. Parse datetime columns
    logger.info("Parsing ScheduledDay and AppointmentDay to datetime")
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

    # 2. Rename columns to snake_case
    logger.info("Renaming columns to snake_case")
    df = df.rename(columns=SNAKE_CASE_MAP)
    logger.info(f"Columns after rename: {list(df.columns)}")

    # 3. Remove duplicate AppointmentIDs
    dup_count = df["appointment_id"].duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates(subset=["appointment_id"], keep="first")
        rows_removed_log.append(f"Duplicates removed: {dup_count:,}")
        logger.info(f"Removed {dup_count:,} duplicate AppointmentIDs")
    else:
        logger.info("No duplicate AppointmentIDs to remove")

    # 4. Filter invalid ages
    invalid_age_mask = (df["age"] < 0) | (df["age"] > 120)
    invalid_age_count = invalid_age_mask.sum()
    if invalid_age_count > 0:
        df = df[~invalid_age_mask]
        rows_removed_log.append(f"Invalid ages (< 0 or > 120): {invalid_age_count:,}")
        logger.info(f"Removed {invalid_age_count:,} records with invalid ages")
    else:
        logger.info("No invalid age records to remove")

    # 5. Filter negative date_diff values
    negative_dd_mask = df["date_diff"] < 0
    negative_dd_count = negative_dd_mask.sum()
    if negative_dd_count > 0:
        df = df[~negative_dd_mask]
        rows_removed_log.append(f"Negative date_diff: {negative_dd_count:,}")
        logger.info(f"Removed {negative_dd_count:,} records with negative date_diff")
    else:
        logger.info("No negative date_diff records to remove")

    # 6. Encode Gender as binary (F=0, M=1)
    logger.info("Encoding Gender: F=0, M=1")
    df["gender"] = df["gender"].map({"F": 0, "M": 1}).astype(int)

    # 7. Convert boolean columns to integer (True/False → 1/0)
    bool_cols_snake = [SNAKE_CASE_MAP[c] for c in BOOLEAN_COLUMNS]
    logger.info(f"Converting boolean columns to int: {bool_cols_snake}")
    for col in bool_cols_snake:
        df[col] = df[col].astype(int)

    # Log removal summary
    final_rows = len(df)
    total_removed = initial_rows - final_rows
    logger.info("-" * 40)
    logger.info(f"SILVER CLEANING SUMMARY:")
    logger.info(f"  Rows in:  {initial_rows:,}")
    logger.info(f"  Rows out: {final_rows:,}")
    logger.info(f"  Removed:  {total_removed:,} ({total_removed / initial_rows * 100:.3f}%)")
    for reason in rows_removed_log:
        logger.info(f"    - {reason}")

    pipeline_summary["silver_rows"] = final_rows

    # Save to silver layer
    os.makedirs("silver", exist_ok=True)
    silver_path = "silver/cleaned_appointments.parquet"
    df.to_parquet(silver_path, index=False)
    logger.info(f"Silver parquet saved: {silver_path}")

    return df


# ──────────────────────────────────────────────
# SILVER VALIDATION
# ──────────────────────────────────────────────

def validate_silver(df: pd.DataFrame, bronze_row_count: int) -> bool:
    """
    Run validation checks on Silver data.
    Returns True if all checks pass.
    """
    logger.info("-" * 60)
    logger.info("SILVER VALIDATION: Running checks on cleaned data")
    logger.info("-" * 60)

    results = {}

    # 1. No nulls remain
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.error(f"NULL CHECK — FAIL: {null_count:,} null values remain")
        results["nulls"] = "FAIL"
    else:
        logger.info("NULL CHECK — PASS: No null values")
        results["nulls"] = "PASS"

    # 2. Data types match expected schema
    expected_dtypes = {
        "patient_id": ["float64"],
        "appointment_id": ["int64"],
        "gender": ["int", "int64", "int32"],
        "scheduled_day": ["datetime64"],
        "appointment_day": ["datetime64"],
        "age": ["int64"],
        "neighbourhood": ["object", "str", "string"],
        "date_diff": ["int64"],
    }
    dtype_issues = []
    for col, accepted in expected_dtypes.items():
        actual = str(df[col].dtype)
        if not any(t in actual for t in accepted):
            dtype_issues.append(f"{col}: expected one of {accepted}, got {actual}")

    if dtype_issues:
        for issue in dtype_issues:
            logger.error(f"DTYPE CHECK — FAIL: {issue}")
        results["dtypes"] = "FAIL"
    else:
        logger.info("DTYPE CHECK — PASS: All columns have expected types post-transform")
        results["dtypes"] = "PASS"

    # 3. Row count within 5% of bronze
    loss_pct = (bronze_row_count - len(df)) / bronze_row_count * 100
    if loss_pct > 5:
        logger.error(f"ROW COUNT CHECK — FAIL: {loss_pct:.2f}% data loss exceeds 5% threshold")
        results["row_count"] = "FAIL"
    else:
        logger.info(f"ROW COUNT CHECK — PASS: {loss_pct:.3f}% data loss (within 5% threshold)")
        results["row_count"] = "PASS"

    # 4. Target variable has both classes
    target_classes = df["showed_up"].unique()
    if len(target_classes) < 2:
        logger.error(f"TARGET CHECK — FAIL: Only {target_classes} found in showed_up")
        results["target"] = "FAIL"
    else:
        class_dist = df["showed_up"].value_counts()
        logger.info(f"TARGET CHECK — PASS: Both classes present (1: {class_dist.get(1, 0):,}, 0: {class_dist.get(0, 0):,})")
        results["target"] = "PASS"

    # Summary
    logger.info("-" * 40)
    logger.info("SILVER VALIDATION SUMMARY:")
    all_passed = True
    for check, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        logger.info(f"  {icon} {check}: {status}")
        if status == "FAIL":
            all_passed = False

    if all_passed:
        logger.info("SILVER VALIDATION: ALL CHECKS PASSED")
        pipeline_summary["silver_validation"] = "PASS"
    else:
        logger.error("SILVER VALIDATION: CHECKS FAILED — Pipeline halted")
        pipeline_summary["silver_validation"] = "FAIL"

    return all_passed


# ══════════════════════════════════════════════
# GOLD LAYER: Feature Engineered + ML-Ready
# ══════════════════════════════════════════════

def engineer_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold Layer: Feature engineering to produce ML-ready dataset.
    """
    logger.info("=" * 60)
    logger.info("GOLD LAYER: Feature Engineering")
    logger.info("=" * 60)

    # 1. lead_time: days between scheduled and appointment
    logger.info("Engineering: lead_time (days between scheduled and appointment)")
    df["lead_time"] = (df["appointment_day"] - df["scheduled_day"]).dt.days

    # 2. day_of_week: from appointment day (0=Monday)
    logger.info("Engineering: day_of_week (0=Monday ... 6=Sunday)")
    df["day_of_week"] = df["appointment_day"].dt.dayofweek

    # 3. is_weekend: binary flag
    logger.info("Engineering: is_weekend (Saturday=1, Sunday=1)")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 4. hour_of_day: from scheduled day timestamp
    logger.info("Engineering: hour_of_day (from scheduled_day timestamp)")
    df["hour_of_day"] = df["scheduled_day"].dt.hour

    # 5. age_group: binned
    logger.info(f"Engineering: age_group (bins: {AGE_LABELS})")
    df["age_group"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).astype(str)

    # 6. chronic_condition_count: sum of chronic indicators
    logger.info("Engineering: chronic_condition_count (hipertension + diabetes + alcoholism + handcap)")
    df["chronic_condition_count"] = (
        df["hipertension"] + df["diabetes"] + df["alcoholism"] + df["handcap"]
    )

    # 7. neighbourhood_noshow_rate: historical avg no-show rate per neighbourhood
    logger.info("Engineering: neighbourhood_noshow_rate (avg no-show rate per neighbourhood)")
    # showed_up=0 means no-show, showed_up=1 means showed up
    # noshow_rate = proportion of no-shows (showed_up == 0)
    noshow_rates = df.groupby("neighbourhood")["showed_up"].apply(lambda x: 1 - x.mean())
    df["neighbourhood_noshow_rate"] = df["neighbourhood"].map(noshow_rates).round(4)

    # Drop raw columns no longer needed for modeling
    # NOTE: patient_id, appointment_id, appointment_day are RETAINED as metadata
    # columns for lookup/display purposes — they must be excluded from model features
    drop_cols = ["scheduled_day", "neighbourhood"]
    logger.info(f"Dropping raw columns: {drop_cols}")
    logger.info("Retaining metadata columns (not model features): patient_id, appointment_id, appointment_day")
    df = df.drop(columns=drop_cols)

    # Log feature summary
    logger.info("-" * 40)
    logger.info(f"GOLD FEATURE SUMMARY:")
    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"  Features: {list(df.columns)}")
    logger.info(f"  Dtypes:\n{df.dtypes.to_string()}")

    pipeline_summary["gold_rows"] = len(df)
    pipeline_summary["gold_features"] = list(df.columns)

    # Save to gold layer
    os.makedirs("gold", exist_ok=True)
    gold_path = "gold/ml_ready_appointments.parquet"
    df.to_parquet(gold_path, index=False)
    logger.info(f"Gold parquet saved: {gold_path}")

    return df


# ──────────────────────────────────────────────
# GOLD VALIDATION
# ──────────────────────────────────────────────

def validate_gold(df: pd.DataFrame) -> bool:
    """
    Run validation checks on Gold (ML-ready) data.
    Returns True if all checks pass.
    """
    logger.info("-" * 60)
    logger.info("GOLD VALIDATION: Running checks on ML-ready data")
    logger.info("-" * 60)

    results = {}

    # Metadata columns (not model features) — skip during feature checks
    METADATA_COLS = ["patient_id", "appointment_id", "appointment_day"]
    feature_df = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])

    # 1. No nulls (check feature columns only)
    null_count = feature_df.isnull().sum().sum()
    if null_count > 0:
        null_breakdown = feature_df.isnull().sum()
        null_cols = null_breakdown[null_breakdown > 0]
        logger.error(f"NULL CHECK — FAIL: {null_count:,} nulls found")
        for col, cnt in null_cols.items():
            logger.error(f"  {col}: {cnt:,} nulls")
        results["nulls"] = "FAIL"
    else:
        logger.info("NULL CHECK — PASS: No null values in final dataset")
        results["nulls"] = "PASS"

    # 2. No infinite values (feature columns only)
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(feature_df[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.error(f"INFINITE CHECK — FAIL: {inf_count:,} infinite values found")
        results["infinite"] = "FAIL"
    else:
        logger.info("INFINITE CHECK — PASS: No infinite values")
        results["infinite"] = "PASS"

    # 3. Feature range checks
    range_issues = []

    if (df["lead_time"] < 0).any():
        neg_lt = (df["lead_time"] < 0).sum()
        range_issues.append(f"lead_time: {neg_lt:,} negative values")

    if not df["day_of_week"].between(0, 6).all():
        range_issues.append("day_of_week: values outside 0-6 range")

    expected_age_groups = set(AGE_LABELS)
    actual_age_groups = set(df["age_group"].unique())
    missing_groups = expected_age_groups - actual_age_groups
    if missing_groups:
        range_issues.append(f"age_group: missing bins: {missing_groups}")

    if range_issues:
        for issue in range_issues:
            logger.warning(f"RANGE CHECK — WARNING: {issue}")
        results["ranges"] = "WARNING"
    else:
        logger.info("RANGE CHECK — PASS: All feature values within expected ranges")
        results["ranges"] = "PASS"

    # 4. Log final dataset info
    logger.info("-" * 40)
    logger.info("FINAL DATASET PROFILE:")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} features")
    logger.info(f"  Features: {list(df.columns)}")

    target_dist = df["showed_up"].value_counts()
    total = len(df)
    logger.info(f"  Target distribution:")
    logger.info(f"    Showed up (1):  {target_dist.get(1, 0):,} ({target_dist.get(1, 0) / total * 100:.1f}%)")
    logger.info(f"    No-show  (0):   {target_dist.get(0, 0):,} ({target_dist.get(0, 0) / total * 100:.1f}%)")

    # Summary
    logger.info("-" * 40)
    logger.info("GOLD VALIDATION SUMMARY:")
    all_passed = True
    for check, status in results.items():
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARNING" else "✗")
        logger.info(f"  {icon} {check}: {status}")
        if status == "FAIL":
            all_passed = False

    if all_passed:
        logger.info("GOLD VALIDATION: ALL CHECKS PASSED")
        pipeline_summary["gold_validation"] = "PASS"
    else:
        logger.error("GOLD VALIDATION: CHECKS FAILED")
        pipeline_summary["gold_validation"] = "FAIL"

    return all_passed


# ══════════════════════════════════════════════
# MAIN: End-to-End Pipeline
# ══════════════════════════════════════════════

def main(csv_path: str = "healthcare_noshows_appt.csv"):
    """Run the full Bronze → Silver → Gold ETL pipeline."""

    start_time = datetime.now()
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   PATIENT NO-SHOW PREDICTION: ETL PIPELINE               ║")
    logger.info("║   Medallion Architecture (Bronze → Silver → Gold)        ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {csv_path}")

    # ── BRONZE ──
    df_bronze = ingest_bronze(csv_path)

    if not validate_bronze(df_bronze):
        logger.error("Pipeline aborted due to Bronze validation failure.")
        return

    # ── SILVER ──
    df_silver = transform_silver(df_bronze)

    if not validate_silver(df_silver, pipeline_summary["bronze_rows"]):
        logger.error("Pipeline aborted due to Silver validation failure.")
        return

    # ── GOLD ──
    df_gold = engineer_gold(df_silver)

    validate_gold(df_gold)

    # ── PIPELINE SUMMARY ──
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Duration:             {duration:.2f} seconds")
    logger.info(f"  Bronze rows:          {pipeline_summary['bronze_rows']:,}")
    logger.info(f"  Silver rows:          {pipeline_summary['silver_rows']:,}")
    logger.info(f"  Gold rows:            {pipeline_summary['gold_rows']:,}")
    logger.info(f"  Total rows removed:   {pipeline_summary['bronze_rows'] - pipeline_summary['gold_rows']:,}")
    logger.info(f"  Bronze validation:    {pipeline_summary['bronze_validation']}")
    logger.info(f"  Silver validation:    {pipeline_summary['silver_validation']}")
    logger.info(f"  Gold validation:      {pipeline_summary['gold_validation']}")
    logger.info(f"  Final features ({len(pipeline_summary['gold_features'])}): {pipeline_summary['gold_features']}")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
