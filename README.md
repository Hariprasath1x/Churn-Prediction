# 📡 Telecom Customer Retention Intelligence System

A professional ML-powered Streamlit dashboard for **churn prediction**, **survival analysis**, and **AI-assisted retention strategy** generation.

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

---

## Project Structure

```
churnprediction/
├── app.py                    # Main Streamlit application
├── models/
│   ├── catboost_churn_model.cbm
│   ├── cox_ph_model.pkl
│   └── logistic_regression_model.pkl
├── data/
│   └── telco_processed.csv
├── utils/
│   ├── __init__.py
│   ├── model_loader.py       # Model loading + sklearn compat patch
│   ├── preprocessing.py      # Feature engineering + Cox OHE
│   ├── prediction.py         # Inference for all 3 models
│   ├── retention_engine.py   # Composite scoring + rule-based recs
│   ├── groq_client.py        # Groq API wrapper
│   └── ai_retention_advisor.py  # AI strategy prompt + parsing
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure Groq API for AI strategies
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

---

## Running the App

```bash
streamlit run app.py
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

### CatBoost (Primary)
- Trained on 23 features with 16 categorical features handled natively
- Used for: churn probability, prediction, feature importance

### Logistic Regression (Baseline)
- sklearn Pipeline with ColumnTransformer (imputation + OHE)
- Used for: comparison baseline only

### Cox PH Survival (lifelines)
- Trained on 35 one-hot-encoded features
- Used for: relative hazard, survival probabilities at 12/24/36 months
- ⚠️ Partial hazard is a **relative** measure — not a churn probability

### Composite Risk Score
```
risk_score = 0.60 × CatBoost_prob + 0.25 × Cox_risk + 0.15 × HighValueCustomer
```

Risk Levels: `LOW` (< 35%) | `MEDIUM` (35–55%) | `HIGH` (55–75%) | `CRITICAL` (≥ 75%)

---

## Groq / AI Strategy

The AI strategy feature uses **Groq's Llama API** to generate personalised retention recommendations.

- Only called when user clicks **"Generate AI Retention Strategy"**
- Never called automatically on Streamlit reruns
- Results cached in `st.session_state`
- Gracefully falls back to rule-based recommendations on any failure

Configure via `.env`:
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Or via Streamlit secrets (deployment):
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_key_here"
```

---

## Batch Prediction

Upload a CSV with the 19 raw customer feature columns. The app will:
1. Auto-compute the 4 engineered features
2. Run CatBoost inference on all rows
3. Apply rule-based retention scoring
4. Show risk distribution and summary stats
5. Allow full CSV download with predictions

---

## Compatibility Notes

- Models were saved with **sklearn 1.6.1** — a compatibility shim in `model_loader.py` patches `_RemainderColsList` for compatibility with newer sklearn versions
- Models require **numpy ≥ 2.0** (saved with numpy 2.x `_core` module)
- CatBoost loads natively via `.load_model()` with no version constraints

---

## License

MIT License — see LICENSE file for details.
