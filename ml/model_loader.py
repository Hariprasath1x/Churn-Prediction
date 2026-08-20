"""
model_loader.py
Loads and caches all ML models. Uses monkey-patching to handle sklearn
version incompatibilities between the saved artifacts (sklearn 1.6.1) and
the current environment.
"""

import warnings
import sys
import joblib
import threading
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


class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._models = {
                    "catboost": None,
                    "lr": None,
                    "cox": None,
                }
                cls._instance._errors = {}
                cls._instance._initialized = False
            return cls._instance

    def initialize(self):
        """Load all models into memory if not already loaded."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
                
            try:
                self._models["catboost"] = self._load_catboost()
            except Exception as e:
                self._errors["catboost"] = str(e)

            try:
                self._models["lr"] = self._load_logistic_regression()
            except Exception as e:
                self._errors["lr"] = str(e)

            try:
                self._models["cox"] = self._load_cox()
            except Exception as e:
                self._errors["cox"] = str(e)
                
            self._initialized = True

    def _load_catboost(self):
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

    def _load_logistic_regression(self):
        """Load the baseline Logistic Regression pipeline."""
        path = MODEL_DIR / "logistic_regression_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"LR model not found at {path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(str(path))
        return model

    def _load_cox(self):
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
        
    def get_catboost_model(self):
        self.initialize()
        if self._errors.get("catboost"):
            raise RuntimeError(f"CatBoost failed to load: {self._errors['catboost']}")
        return self._models["catboost"]
        
    def get_logistic_model(self):
        self.initialize()
        if self._errors.get("lr"):
            raise RuntimeError(f"LR failed to load: {self._errors['lr']}")
        return self._models["lr"]
        
    def get_cox_model(self):
        self.initialize()
        if self._errors.get("cox"):
            raise RuntimeError(f"Cox failed to load: {self._errors['cox']}")
        return self._models["cox"]

    def get_all_models_status(self):
        self.initialize()
        return {
            "catboost_loaded": self._models["catboost"] is not None,
            "lr_loaded": self._models["lr"] is not None,
            "cox_loaded": self._models["cox"] is not None,
            "errors": self._errors
        }

# Global instance for easy access
model_manager = ModelManager()
