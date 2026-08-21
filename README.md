# 📡 Telecom Customer Retention Intelligence System (Django REST API)

A professional ML-powered **Django REST Framework** backend for **churn prediction**, **survival analysis**, and **AI-assisted retention strategy** generation.

This project was recently migrated from a Streamlit monolith to a fully decoupled Django API architecture, featuring thread-safe model loading and scalable Gunicorn deployment.

---

## Features

| Feature | Description |
|---|---|
| **CatBoost Churn Prediction** | Primary churn classifier (23-feature model) |
| **Logistic Regression Baseline** | Comparison baseline model |
| **Cox PH Survival Analysis** | Survival probabilities at 12, 24, 36 months |
| **Composite Risk Scoring** | 60% CatBoost + 25% Cox + 15% Customer Value |
| **Rule-Based Recommendations** | Always-on retention playbook |
| **AI Strategy (Groq/Llama)** | On-demand AI retention consultant |
| **Batch Prediction** | CSV upload → churn + risk scoring for all customers |
| **Report Download** | Plain-text individual analysis reports |
| **OpenAPI Documentation** | Auto-generated Swagger UI (`drf-spectacular`) |

---

## Project Structure

```
churnprediction/
├── config/                   # Django project configuration
├── apps/                     # Django applications
│   ├── predictions/          # Handles single predictions
│   ├── retention/            # Handles AI strategy and reports
│   └── batch/                # Handles bulk CSV predictions
├── ml/                       # Machine Learning logic
│   ├── model_loader.py       # Thread-safe Singleton model loading
│   ├── preprocessing.py      # Feature engineering
│   └── prediction.py         # Inference functions
├── services/                 # External service wrappers
│   ├── retention_engine.py   # Composite scoring & rules
│   ├── groq_client.py        # Groq API client
│   └── ai_retention_advisor.py
├── models/                   # Pre-trained ML models (.pkl, .cbm)
├── data/                     # Sample datasets
├── pyrightconfig.json        # IDE configuration
├── Dockerfile                # Gunicorn production container
├── requirements.txt          # Python dependencies
└── app.py                    # Legacy Streamlit app (kept for reference/dual-run)
```

---

## Installation & Setup

1. **Clone and create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables:**
```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

4. **Run Database Migrations:**
```bash
python manage.py makemigrations
python manage.py migrate
```

---

## Running the API Server

Start the Django development server:
```bash
python manage.py runserver 0.0.0.0:8000
```

Once running, access the interactive API documentation (Swagger UI) at:
**[http://localhost:8000/api/docs/](http://localhost:8000/api/docs/)**

### API Endpoints

- `POST /api/v1/predictions/` : Single customer churn prediction
- `POST /api/v1/customers/ai-strategy/` : Generate AI retention strategy (requires Groq)
- `POST /api/v1/customers/report/` : Generate plain-text analysis report
- `POST /api/v1/batch/predict/` : Upload CSV for batch prediction
- `GET /api/schema/` : Raw OpenAPI YAML schema

*(Note: If you still wish to run the legacy UI, you can run `streamlit run app.py` on port 8501).*

---

## Docker Deployment (Production)

The included `Dockerfile` is optimized for production and serves the Django application using **Gunicorn** with 3 worker processes.

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 --env-file .env churn-prediction-api
```

---

## Engineered Features (auto-computed)

| Feature | Formula |
|---|---|
| `AvgMonthlySpend` | `TotalCharges / tenure` (or `MonthlyCharges` if `tenure == 0`) |
| `NumServices` | Count of "Yes" across 6 add-on service columns |
| `HighValueCustomer` | `1` if `MonthlyCharges > training_median`, else `0` |
| `ContractRisk` | Month-to-month → `High`, One year → `Medium`, Two year → `Low` |

---

## Model Architecture

### Thread-Safe Singleton
Models are loaded via `ModelManager` (in `ml/model_loader.py`). This uses threading locks to guarantee models load only once across Gunicorn worker threads, heavily optimizing memory usage and startup time.

### CatBoost (Primary)
- Trained on 23 features with 16 categorical features handled natively

### Logistic Regression (Baseline)
- sklearn Pipeline with ColumnTransformer (imputation + OHE)

### Cox PH Survival (lifelines)
- Trained on 35 one-hot-encoded features
- ⚠️ Partial hazard is a **relative** measure — not a churn probability

### Composite Risk Score
```
risk_score = 0.60 × CatBoost_prob + 0.25 × Cox_risk + 0.15 × HighValueCustomer
```
Risk Levels: `LOW` (< 35%) | `MEDIUM` (35–55%) | `HIGH` (55–75%) | `CRITICAL` (≥ 75%)

---

## Compatibility Notes

- Models were saved with **sklearn 1.6.1** — a compatibility shim in `ml/model_loader.py` patches `_RemainderColsList` for compatibility with newer sklearn versions.
- Models require **numpy ≥ 2.0** (saved with numpy 2.x `_core` module).
- CatBoost loads natively via `.load_model()` with no version constraints.

---

## License

MIT License — see LICENSE file for details.
