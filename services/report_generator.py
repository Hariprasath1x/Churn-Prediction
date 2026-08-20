import datetime

def generate_report(raw: dict, analysis: dict, ai_strategy: dict | None = None) -> str:
    """Generate a plain-text analysis report for download."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "TELECOM CUSTOMER RETENTION INTELLIGENCE — ANALYSIS REPORT",
        f"Generated: {ts}",
        "=" * 70,
        "",
        "── CUSTOMER PROFILE ──────────────────────────────────────────────",
        f"  Contract:         {raw.get('Contract', 'N/A')}",
        f"  Tenure:           {raw.get('tenure', 0)} months",
        f"  Monthly Charges:  ${raw.get('MonthlyCharges', 0):.2f}",
        f"  Total Charges:    ${raw.get('TotalCharges', 0):.2f}",
        f"  Internet Service: {raw.get('InternetService', 'N/A')}",
        f"  Payment Method:   {raw.get('PaymentMethod', 'N/A')}",
        f"  Senior Citizen:   {'Yes' if raw.get('SeniorCitizen', 0) == 1 else 'No'}",
        f"  Num Add-on Svcs:  {raw.get('NumServices', 0)}",
        "",
        "── ML PREDICTION ─────────────────────────────────────────────────",
        f"  Churn Prediction:      {'WILL CHURN' if analysis.get('churn_prediction') == 1 else 'WILL RETAIN'}",
        f"  CatBoost Probability:  {analysis.get('churn_probability', 0):.1%}",
        f"  LR Baseline Prob:      {analysis.get('lr_probability', 0):.1%}" if analysis.get('lr_probability') is not None else "  LR Baseline:           N/A",
        f"  Risk Level:            {analysis.get('risk_level', 'N/A')}",
        f"  Priority Score:        {analysis.get('priority_score', 0)}/100",
        f"  Annual Revenue:        ${analysis.get('annual_revenue', 0):.2f}",
        "",
        "── SURVIVAL ANALYSIS ─────────────────────────────────────────────",
    ]
    if analysis.get("cox_success"):
        lines += [
            f"  Relative Hazard (Cox): {analysis.get('partial_hazard', 1.0):.4f}",
            f"  Cox Risk Score:        {analysis.get('cox_risk_score', 0.5):.1%}",
            f"  12-Month Survival:     {analysis.get('surv_12', 0.75):.1%}",
            f"  24-Month Survival:     {analysis.get('surv_24', 0.60):.1%}",
            f"  36-Month Survival:     {analysis.get('surv_36', 0.50):.1%}",
        ]
    else:
        lines.append("  Cox model unavailable — survival analysis not performed.")

    lines += [
        "",
        "── RISK FACTORS ──────────────────────────────────────────────────",
    ]
    for f in analysis.get("risk_factors", []):
        lines.append(f"  • {f}")

    rec = analysis.get("recommendation", {})
    lines += [
        "",
        "── RULE-BASED RECOMMENDATION ─────────────────────────────────────",
        f"  {rec.get('summary', 'N/A')}",
        "",
        "  Suggested Actions:",
    ]
    for a in rec.get("actions", []):
        lines.append(f"  → {a}")

    if ai_strategy and ai_strategy.get("ai_generated"):
        lines += [
            "",
            "── AI RETENTION STRATEGY (Groq/Llama) ───────────────────────────",
            f"  Customer Summary:  {ai_strategy.get('customer_summary', 'N/A')}",
            "",
            f"  Risk Analysis:     {ai_strategy.get('risk_analysis', 'N/A')}",
            "",
            "  Recommended Actions:",
        ]
        for a in ai_strategy.get("recommended_actions", []):
            lines.append(f"  → {a}")
        lines += [
            "",
            f"  Personalized Offer:   {ai_strategy.get('personalized_offer', 'N/A')}",
            f"  Communication:        {ai_strategy.get('communication_strategy', 'N/A')}",
            f"  Business Impact:      {ai_strategy.get('expected_business_impact', 'N/A')}",
        ]

    lines += ["", "=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(lines)
