"""
model_loader.py
Loads and caches all ML models. Uses monkey-patching to handle sklearn
version incompatibilities between the saved artifacts (sklearn 1.6.1) and
the current environment.
"""

import warnings
import sys
import joblib
import streamlit as st
from pathlib import Path

# ── sklearn compatibility patch ──────────────────────────────────────────────
# The LR pipeline was saved with sklearn 1.6.x which introduced _RemainderColsList
# inside ColumnTransformer. Older and newer sklearn versions may not have it.
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Compatibility shim for sklearn ColumnTransformer remainder columns."""
        pass
    _ct._RemainderColsList = _RemainderColsList
    sys.modules["sklearn.compose._column_transformer"]._RemainderColsList = _RemainderColsList
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "models"

# ── CatBoost import (optional graceful fail) ─────────────────────────────────
try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

# ── Lifelines import (optional graceful fail) ────────────────────────────────
try:
    import lifelines  # noqa: F401
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False


@st.cache_resource(show_spinner="Loading CatBoost model…")
def load_catboost():
    """Load the primary CatBoost churn classifier."""
    path = MODEL_DIR / "catboost_churn_model.cbm"
    if not path.exists():
        raise FileNotFoundError(f"CatBoost model not found at {path}")
    if not _CATBOOST_AVAILABLE:
        raise ImportError("catboost package not installed. Run: pip install catboost")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CatBoostClassifier()
        model.load_model(str(path))
    return model


@st.cache_resource(show_spinner="Loading Logistic Regression model…")
def load_logistic_regression():
    """Load the baseline Logistic Regression pipeline."""
    path = MODEL_DIR / "logistic_regression_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"LR model not found at {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(str(path))
    return model


@st.cache_resource(show_spinner="Loading Cox PH model…")
def load_cox():
    """Load the Cox Proportional Hazards survival model."""
    path = MODEL_DIR / "cox_ph_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cox model not found at {path}")
    if not _LIFELINES_AVAILABLE:
        raise ImportError("lifelines package not installed. Run: pip install lifelines")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(str(path))
    return model


def load_all_models():
    """
    Attempt to load all models. Returns a dict with model or None per key.
    Errors are captured and returned as strings for UI display.
    """
    results = {
        "catboost": None,
        "lr": None,
        "cox": None,
        "errors": {},
    }

    try:
        results["catboost"] = load_catboost()
    except Exception as e:
        results["errors"]["catboost"] = str(e)

    try:
        results["lr"] = load_logistic_regression()
    except Exception as e:
        results["errors"]["lr"] = str(e)

    try:
        results["cox"] = load_cox()
    except Exception as e:
        results["errors"]["cox"] = str(e)

    return results
