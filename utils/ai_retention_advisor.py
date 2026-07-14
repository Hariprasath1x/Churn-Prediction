"""
ai_retention_advisor.py
Generates AI-powered retention strategies via Groq/Llama.

Only called when user explicitly clicks "Generate AI Retention Strategy".
Results are stored in st.session_state to avoid redundant API calls.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from utils.groq_client import call_groq, is_groq_configured

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert telecom customer retention consultant with 15+ years of experience.
Your role is to provide highly specific, practical, and cost-aware retention strategies.

IMPORTANT GUIDELINES:
- Be specific and personalised — reference the actual customer details provided
- Avoid unnecessary discounts for low-risk customers
- Consider customer lifetime value and cost of acquisition
- Prioritise contract upgrades over discounts where possible
- Be concise but actionable

Respond ONLY with a valid JSON object (no markdown, no code fences) in this exact structure:
{
  "customer_summary": "Brief 2-3 sentence profile of the customer",
  "risk_analysis": "Analysis of why this customer is at risk and key churn drivers",
  "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "personalized_offer": "Specific offer tailored to this customer's profile and usage",
  "communication_strategy": "How, when, and through which channel to reach this customer",
  "expected_business_impact": "Expected outcome if recommendation is followed (revenue, retention probability)"
}"""


def _build_prompt(analysis: dict, raw: dict) -> str:
    """Construct the minimal, structured prompt sent to Groq."""

    surv_12 = analysis.get("surv_12", 0.75)
    surv_24 = analysis.get("surv_24", 0.60)
    surv_36 = analysis.get("surv_36", 0.50)

    rule_rec = analysis.get("recommendation", {})
    rule_summary = rule_rec.get("summary", "N/A")
    rule_actions = rule_rec.get("actions", [])

    prompt = f"""CUSTOMER RETENTION ANALYSIS REQUEST

## Customer Profile
- Contract type: {raw.get('Contract', 'N/A')}
- Tenure: {raw.get('tenure', 0)} months
- Monthly charges: ${raw.get('MonthlyCharges', 0):.2f}
- Annual revenue at risk: ${analysis.get('annual_revenue', 0):.2f}
- Internet service: {raw.get('InternetService', 'N/A')}
- Payment method: {raw.get('PaymentMethod', 'N/A')}
- Senior citizen: {'Yes' if raw.get('SeniorCitizen', 0) == 1 else 'No'}
- Has partner: {raw.get('Partner', 'No')}
- Has dependents: {raw.get('Dependents', 'No')}
- Number of add-on services: {raw.get('NumServices', 0)}
- High-value customer: {'Yes' if raw.get('HighValueCustomer', 0) == 1 else 'No'}

## ML Prediction Results
- Churn probability (CatBoost): {analysis.get('churn_probability', 0):.1%}
- Risk level: {analysis.get('risk_level', 'MEDIUM')}
- Retention priority score: {analysis.get('priority_score', 50)}/100

## Survival Analysis
- 12-month survival probability: {surv_12:.1%}
- 24-month survival probability: {surv_24:.1%}
- 36-month survival probability: {surv_36:.1%}

## Detected Risk Factors
{chr(10).join(f'- {f}' for f in analysis.get('risk_factors', []))}

## Current Rule-Based Recommendation
{rule_summary}
Suggested actions:
{chr(10).join(f'- {a}' for a in rule_actions[:4])}

Please provide a personalised AI retention strategy for this customer."""

    return prompt


def generate_ai_strategy(analysis: dict, raw: dict) -> dict:
    """
    Call Groq API to generate AI retention strategy.
    
    Returns
    -------
    dict with keys: customer_summary, risk_analysis, recommended_actions,
                    personalized_offer, communication_strategy, expected_business_impact,
                    ai_generated (bool), error (str or None)
    """
    if not is_groq_configured():
        return {
            "ai_generated": False,
            "error": "GROQ_API_KEY not configured. Add it to .env file or Streamlit secrets.",
            "fallback_used": True,
        }

    prompt = _build_prompt(analysis, raw)
    raw_response = call_groq(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    if raw_response is None:
        return {
            "ai_generated": False,
            "error": "Groq API call failed or timed out. Please try again later.",
            "fallback_used": True,
        }

    # Parse JSON response
    try:
        # Strip potential markdown wrappers
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        strategy = json.loads(text)
        strategy["ai_generated"] = True
        strategy["error"] = None
        strategy["fallback_used"] = False
        return strategy

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse Groq JSON response: {e}\nResponse: {raw_response[:200]}")
        return {
            "ai_generated": False,
            "error": f"AI response parsing failed: {str(e)[:100]}.",
            "raw_response": raw_response[:500],
            "fallback_used": True,
        }
