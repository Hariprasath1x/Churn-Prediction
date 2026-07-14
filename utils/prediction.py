"""
prediction.py
Model inference functions for CatBoost, Logistic Regression, and Cox PH.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Any

from utils.preprocessing import prepare_cox_input, FEATURE_COLUMNS


# ── CatBoost inference ───────────────────────────────────────────────────────

def predict_catboost(model: Any, df: pd.DataFrame) -> dict:
    """
    Run CatBoost inference. Returns churn probability and prediction.

    Parameters
    ----------
    model : CatBoostClassifier loaded instance
    df    : DataFrame with 23 feature columns (raw, before OHE)
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = model.predict_proba(df)[0]
            churn_proba = float(proba[1])
            prediction = int(churn_proba >= 0.5)
        return {
            "churn_probability": churn_proba,
            "churn_prediction": prediction,
            "success": True,
        }
    except Exception as e:
        return {"churn_probability": 0.5, "churn_prediction": 0, "success": False, "error": str(e)}


def predict_logistic_regression(model: Any, df: pd.DataFrame) -> dict:
    """
    Run Logistic Regression inference (baseline comparison).
    The model is a sklearn Pipeline that handles imputation + OHE internally.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = model.predict_proba(df[FEATURE_COLUMNS])[0]
            churn_proba = float(proba[1])
            prediction = int(churn_proba >= 0.5)
        return {
            "churn_probability": churn_proba,
            "churn_prediction": prediction,
            "success": True,
        }
    except Exception as e:
        return {"churn_probability": 0.5, "churn_prediction": 0, "success": False, "error": str(e)}


# ── Cox PH inference ─────────────────────────────────────────────────────────

def predict_cox(model: Any, df: pd.DataFrame) -> dict:
    """
    Compute Cox PH hazard and survival probabilities.

    Notes
    -----
    - `partial_hazard` is a *relative* hazard (not a probability).
    - Survival probabilities are read from `predict_survival_function` at
      specific time points (months) that fall within the training time range.
    - If a requested month exceeds the maximum baseline_hazard_ time,
      we clamp to the maximum available time.
    """
    try:
        cox_df = prepare_cox_input(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Relative partial hazard
            partial_hazard = float(model.predict_partial_hazard(cox_df).iloc[0])

            # Log hazard (risk score)
            log_hazard = float(np.log(partial_hazard)) if partial_hazard > 0 else 0.0

            # Survival function
            sf = model.predict_survival_function(cox_df)
            times = sf.index.to_numpy()
            survival_vals = sf.iloc[:, 0].to_numpy()

            def get_survival_at(months: int) -> float:
                """Interpolate survival probability at `months`."""
                max_t = times.max()
                t = min(months, max_t)
                if t <= times[0]:
                    return float(survival_vals[0])
                idx = np.searchsorted(times, t)
                if idx >= len(times):
                    return float(survival_vals[-1])
                # Linear interpolation
                t0, t1 = times[max(0, idx - 1)], times[idx]
                s0, s1 = survival_vals[max(0, idx - 1)], survival_vals[idx]
                if t1 == t0:
                    return float(s0)
                return float(s0 + (s1 - s0) * (t - t0) / (t1 - t0))

            surv_12 = get_survival_at(12)
            surv_24 = get_survival_at(24)
            surv_36 = get_survival_at(36)

            # Full survival curve (all time points)
            survival_curve = {
                "times": times.tolist(),
                "survival": survival_vals.tolist(),
            }

        # Normalise hazard to 0–1 risk score (sigmoid-based)
        cox_risk_score = float(1 / (1 + np.exp(-log_hazard * 0.5)))

        return {
            "partial_hazard": partial_hazard,
            "log_hazard": log_hazard,
            "cox_risk_score": cox_risk_score,
            "surv_12": surv_12,
            "surv_24": surv_24,
            "surv_36": surv_36,
            "survival_curve": survival_curve,
            "success": True,
        }

    except Exception as e:
        return {
            "partial_hazard": 1.0,
            "log_hazard": 0.0,
            "cox_risk_score": 0.5,
            "surv_12": 0.75,
            "surv_24": 0.60,
            "surv_36": 0.50,
            "survival_curve": {"times": [], "survival": []},
            "success": False,
            "error": str(e),
        }


# ── Batch inference ──────────────────────────────────────────────────────────

def predict_batch_catboost(model: Any, df: pd.DataFrame) -> pd.Series:
    """Return churn probability for each row in a batch DataFrame."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probas = model.predict_proba(df)[:, 1]
        return pd.Series(probas, index=df.index, name="churn_probability")
    except Exception as e:
        return pd.Series([0.5] * len(df), index=df.index, name="churn_probability")
