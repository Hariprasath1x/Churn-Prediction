"""
preprocessing.py
Handles feature engineering and input preparation for all three models.

Key contracts:
  - LR model  : sklearn Pipeline expecting raw 23-column DataFrame (handles
                 its own imputing + one-hot encoding internally)
  - CatBoost  : expects the same 23 raw columns; cat features passed as strings
  - Cox model : expects a one-hot-encoded 35-column DataFrame matching params_
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

# The exact feature order every model was trained on
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "AvgMonthlySpend", "NumServices", "HighValueCustomer", "ContractRisk",
]

# Categorical features (CatBoost cat_feature_indices → by name)
CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "ContractRisk",
]

# Numeric features
NUM_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "AvgMonthlySpend", "NumServices", "HighValueCustomer",
]

# Cox one-hot encoded column names (from model.params_.index)
COX_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "AvgMonthlySpend", "NumServices", "HighValueCustomer",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "ContractRisk_Low", "ContractRisk_Medium",
]

# Service columns used to compute NumServices
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Contract risk mapping
CONTRACT_RISK_MAP = {
    "Month-to-month": "High",
    "One year": "Medium",
    "Two year": "Low",
}


# Removed st.cache_data for Django compatibility
def get_training_median_monthly_charges() -> float:
    """Return training-set median MonthlyCharges for HighValueCustomer flag."""
    path = DATA_DIR / "telco_processed.csv"
    if path.exists():
        df = pd.read_csv(path)
        return float(df["MonthlyCharges"].median())
    return 64.76  # fallback from the actual dataset


def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Compute derived features from raw customer input and return a
    single-row DataFrame with all 23 model features in the correct order.

    Parameters
    ----------
    raw : dict
        Keys are the original UI-collected fields:
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    """
    tenure = float(raw.get("tenure", 0))
    monthly = float(raw.get("MonthlyCharges", 0.0))
    total = float(raw.get("TotalCharges", 0.0))
    contract = raw.get("Contract", "Month-to-month")

    # ─── Engineered features ──────────────────────────────────────────────
    avg_monthly_spend = (total / tenure) if tenure > 0 else monthly

    num_services = sum(
        1 for col in SERVICE_COLS if raw.get(col, "No") == "Yes"
    )

    training_median = get_training_median_monthly_charges()
    high_value = 1 if monthly > training_median else 0

    contract_risk = CONTRACT_RISK_MAP.get(contract, "High")

    row = {
        # pass-through features
        "gender": raw.get("gender", "Male"),
        "SeniorCitizen": int(raw.get("SeniorCitizen", 0)),
        "Partner": raw.get("Partner", "No"),
        "Dependents": raw.get("Dependents", "No"),
        "tenure": tenure,
        "PhoneService": raw.get("PhoneService", "Yes"),
        "MultipleLines": raw.get("MultipleLines", "No"),
        "InternetService": raw.get("InternetService", "DSL"),
        "OnlineSecurity": raw.get("OnlineSecurity", "No"),
        "OnlineBackup": raw.get("OnlineBackup", "No"),
        "DeviceProtection": raw.get("DeviceProtection", "No"),
        "TechSupport": raw.get("TechSupport", "No"),
        "StreamingTV": raw.get("StreamingTV", "No"),
        "StreamingMovies": raw.get("StreamingMovies", "No"),
        "Contract": contract,
        "PaperlessBilling": raw.get("PaperlessBilling", "No"),
        "PaymentMethod": raw.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        # engineered
        "AvgMonthlySpend": round(avg_monthly_spend, 4),
        "NumServices": num_services,
        "HighValueCustomer": high_value,
        "ContractRisk": contract_risk,
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def prepare_cox_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode a raw feature DataFrame to match the Cox model's
    expected 35-column input (model.params_.index).
    """
    ohe_cats = {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
        "ContractRisk": ["High", "Low", "Medium"],
    }

    result = pd.DataFrame(index=df.index)

    # Numeric columns (pass through)
    for col in NUM_FEATURES:
        result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # One-hot encode, dropping first category (same as drop='first' in OHE)
    for col, cats in ohe_cats.items():
        val = str(df[col].iloc[0]) if len(df) > 0 else cats[0]
        for cat in cats[1:]:  # skip first (dropped reference)
            col_name = f"{col}_{cat}"
            result[col_name] = (df[col] == cat).astype(int)

    # Ensure exact column order matching Cox model params
    cox_df = pd.DataFrame(columns=COX_COLUMNS)
    cox_df = pd.concat([cox_df, result], ignore_index=True)
    cox_df = cox_df[COX_COLUMNS].fillna(0)

    # Ensure numeric dtype
    cox_df = cox_df.astype(float)
    return cox_df


def engineer_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a batch DataFrame (CSV upload).
    Adds AvgMonthlySpend, NumServices, HighValueCustomer, ContractRisk.
    """
    training_median = get_training_median_monthly_charges()

    df = df.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["AvgMonthlySpend"] = df.apply(
        lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
        axis=1,
    )
    df["NumServices"] = df[SERVICE_COLS].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["HighValueCustomer"] = (df["MonthlyCharges"] > training_median).astype(int)
    df["ContractRisk"] = df["Contract"].map(CONTRACT_RISK_MAP).fillna("High")

    return df[FEATURE_COLUMNS]
