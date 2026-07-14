"""
retention_engine.py
Rule-based retention analysis and composite risk scoring.

Risk score formula:
  60% CatBoost churn probability
  25% Cox risk score  (0–1 normalised)
  15% customer value (inverted HighValueCustomer contribution)
"""

from __future__ import annotations

from typing import Any


# ── Risk level thresholds ────────────────────────────────────────────────────

def _classify_risk(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.55:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


# ── Risk factor detection ────────────────────────────────────────────────────

def detect_risk_factors(raw: dict) -> list[str]:
    """Return a list of human-readable risk factors detected for the customer."""
    factors = []

    if raw.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract (highest churn risk)")

    if raw.get("TechSupport") in ("No", "No internet service"):
        factors.append("No technical support subscription")

    if raw.get("OnlineSecurity") in ("No", "No internet service"):
        factors.append("No online security service")

    if raw.get("PaymentMethod") == "Electronic check":
        factors.append("Electronic check payment (associated with higher churn)")

    tenure = float(raw.get("tenure", 0))
    if tenure < 12:
        factors.append(f"Low tenure ({int(tenure)} months) — new customer at risk")

    monthly = float(raw.get("MonthlyCharges", 0))
    if monthly > 80:
        factors.append(f"High monthly charges (${monthly:.2f}) without long-term contract")

    if raw.get("InternetService") == "Fiber optic" and raw.get("Contract") == "Month-to-month":
        factors.append("Fiber optic + month-to-month — premium service without commitment")

    if raw.get("OnlineBackup") in ("No", "No internet service"):
        factors.append("No online backup service")

    if raw.get("SeniorCitizen") == 1:
        factors.append("Senior citizen — may benefit from simplified plan or support")

    num_services = int(raw.get("NumServices", 0))
    if num_services <= 1:
        factors.append(f"Low service adoption ({num_services} add-on services)")

    if not factors:
        factors.append("No major risk factors detected")

    return factors


# ── Rule-based recommendations ───────────────────────────────────────────────

def _rule_based_recommendation(risk_level: str, raw: dict, factors: list[str]) -> dict:
    """Generate structured rule-based retention recommendation."""

    contract = raw.get("Contract", "Month-to-month")
    monthly = float(raw.get("MonthlyCharges", 0))
    tenure = float(raw.get("tenure", 0))
    internet = raw.get("InternetService", "DSL")
    payment = raw.get("PaymentMethod", "")

    actions = []
    summary = ""

    if risk_level == "CRITICAL":
        summary = "Immediate intervention required. Customer is at very high risk of churning."
        actions = [
            "Assign dedicated account manager for immediate personal outreach",
            "Offer a 2-year contract upgrade with 20-25% discount on monthly charges",
            "Bundle TechSupport and OnlineSecurity at no additional cost for 6 months",
            "Escalate to retention specialist team within 24 hours",
        ]
        if payment == "Electronic check":
            actions.append("Offer automatic payment discount (3-5%) to switch payment method")
        if internet == "Fiber optic":
            actions.append("Provide speed upgrade or equipment upgrade at no extra cost")

    elif risk_level == "HIGH":
        summary = "Proactive intervention recommended. Significant churn signals detected."
        actions = [
            "Reach out within 48 hours via preferred communication channel",
            "Offer contract upgrade from month-to-month to 1-year with 15% discount",
            "Provide free 3-month trial of TechSupport and OnlineSecurity",
        ]
        if tenure < 12:
            actions.append("Assign loyalty onboarding program for new customers")
        if monthly > 70:
            actions.append("Review billing for eligible plan optimisation or bundle discount")

    elif risk_level == "MEDIUM":
        summary = "Monitor and engage. Customer shows moderate churn risk signals."
        actions = [
            "Send personalised loyalty offer email within 1 week",
            "Offer 10% loyalty discount for committing to annual contract",
            "Highlight value of unused services (OnlineSecurity, OnlineBackup)",
        ]
        if num_services := int(raw.get("NumServices", 0)) <= 2:
            actions.append("Recommend service bundle upgrade for better value")

    else:  # LOW
        summary = "Customer appears stable. Maintain satisfaction and deepen engagement."
        actions = [
            "Send quarterly satisfaction survey",
            "Offer loyalty reward points or referral bonus",
            "Recommend premium services as upsell (StreamingTV, StreamingMovies)",
        ]

    if contract == "Month-to-month" and risk_level in ("CRITICAL", "HIGH"):
        actions.insert(1, "Priority: Convert to annual/biennial contract to lock in revenue")

    return {
        "summary": summary,
        "actions": actions,
        "rule_based": True,
    }


# ── Composite scoring and retention analysis ─────────────────────────────────

def compute_retention_analysis(
    raw: dict,
    cb_result: dict,
    cox_result: dict,
    lr_result: dict | None = None,
) -> dict:
    """
    Combine model outputs into a unified retention intelligence report.

    Returns
    -------
    dict with keys:
        churn_prediction, churn_probability, lr_probability,
        risk_level, priority_score (0–100),
        annual_revenue, cox_* survival fields,
        risk_factors, recommendation
    """

    cb_prob = cb_result.get("churn_probability", 0.5)
    cox_risk = cox_result.get("cox_risk_score", 0.5)
    high_value = int(raw.get("HighValueCustomer", 0))

    # Customer value component: high-value customers get urgency boost
    # value_weight: 0 = low value, 1 = high value (we ADD risk for high value)
    value_weight = high_value  # 1 if high-value, 0 otherwise

    # Composite risk score
    composite = (0.60 * cb_prob) + (0.25 * cox_risk) + (0.15 * value_weight)
    composite = float(min(max(composite, 0.0), 1.0))

    priority_score = int(composite * 100)
    risk_level = _classify_risk(composite)

    churn_prediction = int(cb_prob >= 0.5)
    annual_revenue = float(raw.get("MonthlyCharges", 0)) * 12

    risk_factors = detect_risk_factors(raw)
    recommendation = _rule_based_recommendation(risk_level, raw, risk_factors)

    return {
        # Core predictions
        "churn_prediction": churn_prediction,
        "churn_probability": cb_prob,
        "lr_probability": lr_result.get("churn_probability") if lr_result else None,
        # Risk
        "composite_score": composite,
        "risk_level": risk_level,
        "priority_score": priority_score,
        # Revenue
        "annual_revenue": annual_revenue,
        # Cox survival
        "partial_hazard": cox_result.get("partial_hazard", 1.0),
        "cox_risk_score": cox_risk,
        "surv_12": cox_result.get("surv_12", 0.75),
        "surv_24": cox_result.get("surv_24", 0.60),
        "surv_36": cox_result.get("surv_36", 0.50),
        "survival_curve": cox_result.get("survival_curve", {}),
        "cox_success": cox_result.get("success", False),
        # Explainability
        "risk_factors": risk_factors,
        # Recommendation
        "recommendation": recommendation,
    }


def compute_batch_retention(df_raw, cb_probabilities) -> "pd.DataFrame":
    """Apply rule-based retention logic to a batch of customers."""
    import pandas as pd

    df = df_raw.copy()
    df["churn_probability"] = cb_probabilities.values
    df["churn_prediction"] = (df["churn_probability"] >= 0.5).astype(int)

    def row_to_risk(row):
        cb = float(row["churn_probability"])
        hv = int(row.get("HighValueCustomer", 0))
        composite = (0.60 * cb) + (0.15 * hv)
        composite = min(max(composite, 0.0), 1.0)
        return _classify_risk(composite)

    def row_to_priority(row):
        cb = float(row["churn_probability"])
        hv = int(row.get("HighValueCustomer", 0))
        composite = (0.60 * cb) + (0.15 * hv)
        return int(min(max(composite, 0.0), 1.0) * 100)

    def row_to_action(row):
        risk = row["risk_level"]
        contract = row.get("Contract", "")
        if risk == "CRITICAL":
            return "Immediate intervention — assign retention specialist and offer contract upgrade"
        elif risk == "HIGH":
            return "Proactive outreach — offer 15% discount for annual contract within 48h"
        elif risk == "MEDIUM":
            return "Monitor and send personalised loyalty offer email"
        return "Maintain satisfaction — include in loyalty reward programme"

    df["risk_level"] = df.apply(row_to_risk, axis=1)
    df["priority_score"] = df.apply(row_to_priority, axis=1)
    df["recommended_action"] = df.apply(row_to_action, axis=1)

    return df
