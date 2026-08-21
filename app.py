"""
app.py — Telecom Customer Retention Intelligence System
Streamlit dashboard for ML-powered churn prediction, survival analysis,
and AI-assisted retention strategy generation.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import io
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ml.model_loader import model_manager
from ml.preprocessing import engineer_features, engineer_batch, FEATURE_COLUMNS
from ml.prediction import predict_catboost, predict_logistic_regression, predict_cox, predict_batch_catboost
from services.retention_engine import compute_retention_analysis, compute_batch_retention
from services.ai_retention_advisor import generate_ai_strategy
from services.groq_client import is_groq_configured

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Customer Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #0a1628 100%);
    min-height: 100vh;
  }

  /* Header */
  .main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
  }
  .main-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
  }
  .main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, rgba(15,22,45,0.9), rgba(20,30,60,0.9));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .metric-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.5); }
  .metric-label { color: #94a3b8; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
  .metric-value { color: #e2e8f0; font-size: 2rem; font-weight: 700; line-height: 1.1; }
  .metric-value.critical { color: #f87171; }
  .metric-value.high { color: #fb923c; }
  .metric-value.medium { color: #fbbf24; }
  .metric-value.low { color: #4ade80; }

  /* Risk badges */
  .risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.06em;
  }
  .badge-critical { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
  .badge-high { background: rgba(249,115,22,0.2); color: #fb923c; border: 1px solid rgba(249,115,22,0.4); }
  .badge-medium { background: rgba(234,179,8,0.2); color: #fbbf24; border: 1px solid rgba(234,179,8,0.4); }
  .badge-low { background: rgba(74,222,128,0.2); color: #4ade80; border: 1px solid rgba(74,222,128,0.4); }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(15,22,45,0.8); border-radius: 12px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #94a3b8; font-weight: 500; padding: 0.5rem 1.2rem; }
  .stTabs [aria-selected="true"] { background: rgba(99,102,241,0.25); color: #818cf8; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 100%); border-right: 1px solid rgba(99,102,241,0.15); }
  [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
    transform: translateY(-1px);
  }

  /* Section headers */
  .section-header {
    color: #818cf8;
    font-size: 1.05rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* Info boxes */
  .info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
  }
  .warning-box {
    background: rgba(234,179,8,0.08);
    border: 1px solid rgba(234,179,8,0.2);
    border-left: 4px solid #eab308;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
  }
  .success-box {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    border-left: 4px solid #4ade80;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
  }
  .danger-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
  }

  /* Factor pills */
  .factor-pill {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 50px;
    padding: 0.25rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem;
  }

  /* Action items */
  .action-item {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    color: #e2e8f0;
    font-size: 0.9rem;
  }
  .action-item::before { content: "→ "; color: #818cf8; font-weight: 700; }

  /* Probability bar */
  .prob-bar-container { background: rgba(30,41,59,0.5); border-radius: 50px; height: 10px; margin: 0.5rem 0; }
  .prob-bar { height: 10px; border-radius: 50px; transition: width 0.8s ease; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_color(level: str) -> str:
    return {"CRITICAL": "#f87171", "HIGH": "#fb923c", "MEDIUM": "#fbbf24", "LOW": "#4ade80"}.get(level, "#94a3b8")

def risk_badge_class(level: str) -> str:
    return {"CRITICAL": "badge-critical", "HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "badge-low")

def metric_value_class(level: str) -> str:
    return {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(level, "low")

def prob_bar_color(prob: float) -> str:
    if prob >= 0.7: return "linear-gradient(90deg, #ef4444, #f87171)"
    if prob >= 0.5: return "linear-gradient(90deg, #f97316, #fb923c)"
    if prob >= 0.3: return "linear-gradient(90deg, #eab308, #fbbf24)"
    return "linear-gradient(90deg, #22c55e, #4ade80)"

def render_metric_card(label: str, value: str, extra_class: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {extra_class}">{value}</div>
    </div>"""


# ── Plotly charts ─────────────────────────────────────────────────────────────

def make_gauge(prob: float, risk_level: str) -> go.Figure:
    color = risk_color(risk_level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Inter"}},
        delta={"reference": 50, "suffix": "%", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569",
                     "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(74,222,128,0.15)"},
                {"range": [35, 55], "color": "rgba(234,179,8,0.15)"},
                {"range": [55, 75], "color": "rgba(249,115,22,0.15)"},
                {"range": [75, 100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
        title={"text": f"Churn Risk<br><span style='font-size:0.9em;color:#94a3b8'>{risk_level}</span>",
               "font": {"size": 16, "color": "#e2e8f0", "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=60, b=0, l=30, r=30),
        font={"family": "Inter"},
    )
    return fig


def make_survival_curve(curve_data: dict) -> go.Figure:
    times = curve_data.get("times", [])
    survival = curve_data.get("survival", [])

    fig = go.Figure()

    if times and survival:
        # Fill area
        fig.add_trace(go.Scatter(
            x=times, y=survival,
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            line=dict(color="#818cf8", width=2.5),
            name="Survival Probability",
            hovertemplate="Month %{x}<br>Survival: %{y:.1%}<extra></extra>",
        ))

        # Mark 12, 24, 36 months
        for t, label in [(12, "12m"), (24, "24m"), (36, "36m")]:
            if times and max(times) >= t:
                idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
                s_val = survival[idx]
                fig.add_trace(go.Scatter(
                    x=[t], y=[s_val],
                    mode="markers+text",
                    marker=dict(size=10, color="#a78bfa", line=dict(color="#fff", width=2)),
                    text=[f"{label}: {s_val:.0%}"],
                    textposition="top center",
                    textfont=dict(color="#c4b5fd", size=11),
                    showlegend=False,
                    hovertemplate=f"Month {t}<br>Survival: {s_val:.1%}<extra></extra>",
                ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,22,45,0.4)",
        height=340,
        margin=dict(t=20, b=50, l=60, r=20),
        xaxis=dict(
            title="Time (Months)", color="#94a3b8",
            gridcolor="rgba(99,102,241,0.1)", zeroline=False,
        ),
        yaxis=dict(
            title="Survival Probability", color="#94a3b8",
            gridcolor="rgba(99,102,241,0.1)", zeroline=False,
            tickformat=".0%", range=[0, 1.05],
        ),
        font={"family": "Inter", "color": "#94a3b8"},
        showlegend=False,
    )
    return fig


def make_feature_importance(model) -> go.Figure | None:
    try:
        importances = model.get_feature_importance()
        feat_names = model.feature_names_
        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=True).tail(15)

        colors = [
            f"rgba(99,102,241,{0.4 + 0.6 * (i / len(df_imp))})"
            for i in range(len(df_imp))
        ]

        fig = go.Figure(go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,45,0.4)",
            height=420,
            margin=dict(t=20, b=40, l=10, r=20),
            xaxis=dict(title="Importance Score", color="#94a3b8",
                       gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(color="#e2e8f0"),
            font={"family": "Inter", "color": "#94a3b8"},
        )
        return fig
    except Exception:
        return None


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(raw: dict, analysis: dict, ai_strategy: dict | None) -> str:
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


def _render_ai_strategy(ai_strat: dict):
    """Render the AI strategy dict as structured Streamlit elements."""
    if ai_strat.get("customer_summary"):
        st.markdown(f"""
        <div class="info-box">
            <strong>Customer Summary</strong><br>{ai_strat['customer_summary']}
        </div>""", unsafe_allow_html=True)

    if ai_strat.get("risk_analysis"):
        st.markdown(f"""
        <div class="warning-box">
            <strong>Risk Analysis</strong><br>{ai_strat['risk_analysis']}
        </div>""", unsafe_allow_html=True)

    if ai_strat.get("recommended_actions"):
        st.markdown("**AI Recommended Actions:**")
        for a in ai_strat["recommended_actions"]:
            st.markdown(f'<div class="action-item">{a}</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    if ai_strat.get("personalized_offer"):
        with cols[0]:
            st.markdown(f"""
            <div class="success-box">
                <strong>🎁 Personalised Offer</strong><br>{ai_strat['personalized_offer']}
            </div>""", unsafe_allow_html=True)

    if ai_strat.get("communication_strategy"):
        with cols[1]:
            st.markdown(f"""
            <div class="info-box">
                <strong>📣 Communication Strategy</strong><br>{ai_strat['communication_strategy']}
            </div>""", unsafe_allow_html=True)

    if ai_strat.get("expected_business_impact"):
        st.markdown(f"""
        <div class="success-box">
            <strong>💰 Expected Business Impact</strong><br>{ai_strat['expected_business_impact']}
        </div>""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "raw_input" not in st.session_state:
    st.session_state.raw_input = None
if "ai_strategy" not in st.session_state:
    st.session_state.ai_strategy = None
if "df_features" not in st.session_state:
    st.session_state.df_features = None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Telecom Customer Retention Intelligence</h1>
    <p>ML-powered churn prediction, survival analysis, and AI-assisted retention strategy</p>
</div>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Initialising models…"):
    model_manager.initialize()
    models = {
        "catboost": model_manager.models.get("catboost"),
        "lr": model_manager.models.get("lr"),
        "cox": model_manager.models.get("cox"),
        "errors": model_manager.errors
    }

cb_model = models["catboost"]
lr_model = models["lr"]
cox_model = models["cox"]
model_errors = models["errors"]

# Show model status in sidebar
with st.sidebar:
    st.markdown("### 🤖 Model Status")
    for name, label in [("catboost", "CatBoost (Primary)"), ("lr", "Logistic Regression"), ("cox", "Cox PH Survival")]:
        if models[name] is not None:
            st.markdown(f"✅ **{label}**")
        else:
            err = model_errors.get(name, "Unknown error")
            st.markdown(f"❌ **{label}**")
            with st.expander(f"Error details"):
                st.code(err, language=None)

    st.markdown("---")
    groq_ok = is_groq_configured()
    if groq_ok:
        st.markdown("🔑 **AI Features**: Active")
    else:
        st.markdown("⚠️ **AI Features**: Unavailable")
        st.caption("AI strategy generation is currently offline.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This dashboard combines CatBoost churn prediction, "
        "Cox Proportional Hazards survival analysis, and "
        "Groq/Llama AI retention strategies."
    )

if cb_model is None:
    st.error("⛔ CatBoost model failed to load. Cannot continue without the primary model.")
    st.stop()


# ── Main tabs ─────────────────────────────────────────────────────────────────
main_tab, batch_tab = st.tabs(["🎯 Individual Analysis", "📊 Batch Prediction"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with main_tab:

    # ── Customer input form ───────────────────────────────────────────────────
    with st.expander("📋 Customer Input Form", expanded=True):
        st.markdown('<div class="section-header">Customer Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        with col2:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior")
        with col3:
            partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        with col4:
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")

        st.markdown('<div class="section-header">Account Information</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
        with col6:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        with col7:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")

        col8, col9 = st.columns(2)
        with col8:
            payment = st.selectbox("Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment")
        with col9:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5, key="monthly")

        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                         float(tenure * monthly), 1.0, key="total")

        st.markdown('<div class="section-header">Phone & Internet Services</div>', unsafe_allow_html=True)
        col10, col11, col12 = st.columns(3)
        with col10:
            phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        with col11:
            multi_lines = st.selectbox("Multiple Lines",
                ["No", "Yes", "No phone service"], key="multilines")
        with col12:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")

        st.markdown('<div class="section-header">Add-on Services</div>', unsafe_allow_html=True)
        col13, col14, col15 = st.columns(3)
        with col13:
            online_sec = st.selectbox("Online Security",
                ["No", "Yes", "No internet service"], key="online_sec")
            device_prot = st.selectbox("Device Protection",
                ["No", "Yes", "No internet service"], key="device_prot")
        with col14:
            online_bak = st.selectbox("Online Backup",
                ["No", "Yes", "No internet service"], key="online_bak")
            streaming_tv = st.selectbox("Streaming TV",
                ["No", "Yes", "No internet service"], key="streaming_tv")
        with col15:
            tech_sup = st.selectbox("Tech Support",
                ["No", "Yes", "No internet service"], key="tech_sup")
            streaming_mov = st.selectbox("Streaming Movies",
                ["No", "Yes", "No internet service"], key="streaming_mov")

    # ── Analyze button ────────────────────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze Customer Risk", key="analyze_btn", use_container_width=True)

    if analyze_clicked:
        raw = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }

        with st.spinner("Running ML inference…"):
            df_features = engineer_features(raw)

            # Add engineered features back to raw for retention engine
            raw["AvgMonthlySpend"] = float(df_features["AvgMonthlySpend"].iloc[0])
            raw["NumServices"] = int(df_features["NumServices"].iloc[0])
            raw["HighValueCustomer"] = int(df_features["HighValueCustomer"].iloc[0])
            raw["ContractRisk"] = str(df_features["ContractRisk"].iloc[0])

            cb_result = predict_catboost(cb_model, df_features)
            lr_result = predict_logistic_regression(lr_model, df_features) if lr_model else None
            cox_result = predict_cox(cox_model, df_features) if cox_model else {
                "partial_hazard": 1.0, "cox_risk_score": 0.5,
                "surv_12": 0.75, "surv_24": 0.60, "surv_36": 0.50,
                "survival_curve": {}, "success": False,
                "error": "Cox model not loaded",
            }

            analysis = compute_retention_analysis(raw, cb_result, cox_result, lr_result)
            st.session_state.analysis_result = analysis
            st.session_state.raw_input = raw
            st.session_state.df_features = df_features
            st.session_state.ai_strategy = None  # reset on new analysis

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.analysis_result is not None:
        analysis = st.session_state.analysis_result
        raw = st.session_state.raw_input
        risk_level = analysis["risk_level"]

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Risk Overview", "📈 Survival Analysis",
            "🔍 Explainability", "💡 Retention Strategy"
        ])

        # ── Tab 1: Risk Overview ──────────────────────────────────────────────
        with tab1:
            st.markdown("#### Customer Risk Overview")

            col_gauge, col_metrics = st.columns([1, 1.5])

            with col_gauge:
                fig_gauge = make_gauge(analysis["churn_probability"], risk_level)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_metrics:
                # Prediction badge
                pred_text = "⚠️ PREDICTED TO CHURN" if analysis["churn_prediction"] == 1 else "✅ PREDICTED TO RETAIN"
                pred_color = "#f87171" if analysis["churn_prediction"] == 1 else "#4ade80"
                st.markdown(f"""
                <div style="background:rgba(15,22,45,0.9);border:1px solid {pred_color}33;
                     border-left:4px solid {pred_color};border-radius:12px;
                     padding:1rem 1.4rem;margin-bottom:1rem">
                    <div style="color:{pred_color};font-weight:700;font-size:1.1rem">{pred_text}</div>
                </div>""", unsafe_allow_html=True)

                # Key metrics grid
                m1, m2 = st.columns(2)
                with m1:
                    cb_prob = analysis["churn_probability"]
                    st.markdown(render_metric_card("CatBoost Churn Prob",
                        f"{cb_prob:.1%}", metric_value_class(risk_level)), unsafe_allow_html=True)
                with m2:
                    lr_prob = analysis.get("lr_probability")
                    if lr_prob is not None:
                        st.markdown(render_metric_card("LR Baseline Prob",
                            f"{lr_prob:.1%}"), unsafe_allow_html=True)
                    else:
                        st.markdown(render_metric_card("LR Baseline", "N/A"), unsafe_allow_html=True)

                m3, m4 = st.columns(2)
                with m3:
                    st.markdown(render_metric_card("Priority Score",
                        f"{analysis['priority_score']}/100",
                        metric_value_class(risk_level)), unsafe_allow_html=True)
                with m4:
                    st.markdown(render_metric_card("Annual Revenue",
                        f"${analysis['annual_revenue']:,.0f}"), unsafe_allow_html=True)

                # Risk level badge
                badge_cls = risk_badge_class(risk_level)
                st.markdown(f"""
                <div style="text-align:center;margin-top:1rem">
                    <span style="color:#94a3b8;font-size:0.85rem">Risk Classification: </span>
                    <span class="risk-badge {badge_cls}">{risk_level}</span>
                </div>""", unsafe_allow_html=True)

            # Probability comparison bars
            st.markdown("#### Model Probability Comparison")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                cb_pct = int(analysis["churn_probability"] * 100)
                col = prob_bar_color(analysis["churn_probability"])
                st.markdown(f"""
                <div style="margin-bottom:0.8rem">
                    <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.3rem">
                        CatBoost (Primary) — {cb_pct}%
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width:{cb_pct}%;background:{col}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_b2:
                if analysis.get("lr_probability") is not None:
                    lr_pct = int(analysis["lr_probability"] * 100)
                    col = prob_bar_color(analysis["lr_probability"])
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem">
                        <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.3rem">
                            Logistic Regression (Baseline) — {lr_pct}%
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width:{lr_pct}%;background:{col}"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Engineered features summary
            with st.expander("🔧 Auto-Generated Engineered Features"):
                df_feat = st.session_state.df_features
                if df_feat is not None:
                    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
                    ecol1.metric("Avg Monthly Spend", f"${float(df_feat['AvgMonthlySpend'].iloc[0]):.2f}")
                    ecol2.metric("Num Services", int(df_feat['NumServices'].iloc[0]))
                    ecol3.metric("High-Value Customer", "Yes" if int(df_feat['HighValueCustomer'].iloc[0]) == 1 else "No")
                    ecol4.metric("Contract Risk", str(df_feat['ContractRisk'].iloc[0]))

        # ── Tab 2: Survival Analysis ──────────────────────────────────────────
        with tab2:
            st.markdown("#### Survival Analysis — Cox Proportional Hazards")

            if not analysis.get("cox_success"):
                err = model_errors.get("cox", "Cox model not loaded") if not cox_model else "Inference error"
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>Cox PH model inference unavailable</strong><br>
                    {err}<br><br>
                    Churn probability and retention score are still fully functional via CatBoost.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    ℹ️ <strong>Note on Cox Hazard:</strong>
                    The <em>partial hazard</em> is a <strong>relative</strong> measure — a value above 1.0
                    means higher hazard relative to the baseline cohort. It is <strong>not</strong> a
                    churn probability. Survival probabilities (below) are the direct retention estimates.
                </div>""", unsafe_allow_html=True)

                # Cox metrics
                cox_cols = st.columns(5)
                cox_cols[0].metric("Relative Hazard", f"{analysis['partial_hazard']:.4f}")
                cox_cols[1].metric("Cox Risk Score", f"{analysis['cox_risk_score']:.1%}")
                cox_cols[2].metric("12-Mo Survival", f"{analysis['surv_12']:.1%}")
                cox_cols[3].metric("24-Mo Survival", f"{analysis['surv_24']:.1%}")
                cox_cols[4].metric("36-Mo Survival", f"{analysis['surv_36']:.1%}")

                # Survival curve
                st.markdown("#### Survival Probability Curve")
                curve = analysis.get("survival_curve", {})
                if curve.get("times"):
                    fig_surv = make_survival_curve(curve)
                    st.plotly_chart(fig_surv, use_container_width=True)
                else:
                    st.info("Survival curve data unavailable.")

                # Survival probability table
                surv_data = {
                    "Time Horizon": ["12 months", "24 months", "36 months"],
                    "Survival Probability": [
                        f"{analysis['surv_12']:.1%}",
                        f"{analysis['surv_24']:.1%}",
                        f"{analysis['surv_36']:.1%}",
                    ],
                    "Churn Risk": [
                        f"{1 - analysis['surv_12']:.1%}",
                        f"{1 - analysis['surv_24']:.1%}",
                        f"{1 - analysis['surv_36']:.1%}",
                    ],
                }
                st.dataframe(pd.DataFrame(surv_data), use_container_width=True, hide_index=True)

        # ── Tab 3: Explainability ─────────────────────────────────────────────
        with tab3:
            st.markdown("#### Customer-Specific Risk Factors")
            st.markdown("""
            <div class="info-box">
                🔍 <strong>Customer-Specific Explanation</strong>: These factors were detected
                from <em>this customer's profile</em> using domain rules. They highlight
                the specific attributes driving churn risk for this individual.
            </div>""", unsafe_allow_html=True)

            factors = analysis.get("risk_factors", [])
            pills_html = "".join(f'<span class="factor-pill">⚠ {f}</span>' for f in factors)
            st.markdown(f'<div style="line-height:2.5">{pills_html}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Global CatBoost Feature Importance")
            st.markdown("""
            <div class="info-box">
                📊 <strong>Global Feature Importance</strong>: This chart shows which features
                matter most to the CatBoost model <em>across all customers</em> in the training
                data — not specific to this individual customer.
            </div>""", unsafe_allow_html=True)

            fig_imp = make_feature_importance(cb_model)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance chart unavailable.")

        # ── Tab 4: Retention Strategy ─────────────────────────────────────────
        with tab4:
            st.markdown("#### Retention Strategy")

            # Rule-based recommendation (always shown)
            rec = analysis.get("recommendation", {})
            risk_c = risk_color(risk_level)
            st.markdown(f"""
            <div style="background:rgba(15,22,45,0.9);border:1px solid {risk_c}33;
                 border-left:4px solid {risk_c};border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.2rem">
                <div style="color:{risk_c};font-weight:700;font-size:1rem;margin-bottom:0.5rem">
                    📋 Rule-Based Recommendation
                </div>
                <div style="color:#e2e8f0;font-size:0.95rem">{rec.get('summary', '')}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Suggested Actions:**")
            for action in rec.get("actions", []):
                st.markdown(f'<div class="action-item">{action}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # AI strategy section
            st.markdown("#### 🤖 AI-Powered Retention Strategy (Groq/Llama)")

            ai_strat = st.session_state.ai_strategy
            if ai_strat and ai_strat.get("ai_generated"):
                st.markdown('<div class="success-box">✅ AI strategy generated successfully.</div>', unsafe_allow_html=True)
                _render_ai_strategy(ai_strat)
            else:
                if ai_strat and not ai_strat.get("ai_generated"):
                    err_msg = ai_strat.get("error", "Unknown error")
                    st.markdown(f'<div class="warning-box">⚠️ {err_msg}</div>', unsafe_allow_html=True)

                if not is_groq_configured():
                    st.markdown("""
                    <div class="info-box">
                        💡 <strong>AI Strategy Generation</strong><br>
                        The AI-powered strategy generation feature is currently unavailable. Please check back later.
                    </div>""", unsafe_allow_html=True)
                else:
                    col_ai1, col_ai2, col_ai3 = st.columns([1, 2, 1])
                    with col_ai2:
                        if st.button("🤖 Generate AI Retention Strategy", key="gen_ai_btn", use_container_width=True):
                            with st.spinner("Consulting Llama AI for personalised retention strategy…"):
                                ai_result = generate_ai_strategy(analysis, raw)
                                st.session_state.ai_strategy = ai_result
                            st.rerun()

        # ── Download report ───────────────────────────────────────────────────
        st.markdown("---")
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            report_text = generate_report(raw, analysis, st.session_state.ai_strategy)
            st.download_button(
                "📄 Download Analysis Report",
                data=report_text,
                file_name=f"retention_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )





# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with batch_tab:
    st.markdown("#### 📊 Batch Customer Churn Prediction")
    st.markdown("""
    <div class="info-box">
        Upload a CSV file containing customer records. Required columns:<br>
        <code>gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges</code>
        <br><br>
        Engineered features (AvgMonthlySpend, NumServices, HighValueCustomer, ContractRisk)
        will be automatically computed.
    </div>""", unsafe_allow_html=True)

    required_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
    ]

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"], key="batch_upload")

    if uploaded:
        try:
            df_raw_upload = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_raw_upload):,} rows** loaded. Preview:")
            st.dataframe(df_raw_upload.head(5), use_container_width=True)

            # Validate columns
            missing_cols = [c for c in required_cols if c not in df_raw_upload.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                with st.spinner(f"Running batch prediction on {len(df_raw_upload):,} customers…"):
                    df_engineered = engineer_batch(df_raw_upload)
                    cb_probs = predict_batch_catboost(cb_model, df_engineered)
                    df_results = compute_batch_retention(df_engineered, cb_probs)

                    # Select output columns
                    output_cols = FEATURE_COLUMNS + [
                        "churn_probability", "churn_prediction",
                        "risk_level", "priority_score", "recommended_action",
                    ]
                    output_cols = [c for c in output_cols if c in df_results.columns]
                    df_output = df_results[output_cols].copy()
                    df_output["churn_probability"] = df_output["churn_probability"].round(4)

                st.success(f"✅ Batch prediction complete for {len(df_output):,} customers.")

                # Summary statistics
                st.markdown("#### Risk Level Distribution")
                risk_counts = df_output["risk_level"].value_counts()
                fig_risk = px.bar(
                    x=risk_counts.index, y=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_map={
                        "CRITICAL": "#f87171", "HIGH": "#fb923c",
                        "MEDIUM": "#fbbf24", "LOW": "#4ade80",
                    },
                    labels={"x": "Risk Level", "y": "Count"},
                )
                fig_risk.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,22,45,0.4)",
                    height=250,
                    margin=dict(t=10, b=40, l=40, r=10),
                    font={"family": "Inter", "color": "#94a3b8"},
                    showlegend=False,
                )
                st.plotly_chart(fig_risk, use_container_width=True)

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Total Customers", f"{len(df_output):,}")
                col_s2.metric("Predicted to Churn", f"{(df_output['churn_prediction'] == 1).sum():,}")
                col_s3.metric("Critical Risk", f"{(df_output['risk_level'] == 'CRITICAL').sum():,}")
                col_s4.metric("Avg Churn Probability", f"{df_output['churn_probability'].mean():.1%}")

                # Preview results
                st.markdown("#### Results Preview")
                st.dataframe(df_output.head(20), use_container_width=True, hide_index=True)

                # Download
                csv_buffer = io.StringIO()
                df_output.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"batch_churn_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#475569;font-size:0.8rem">
    Telecom Customer Retention Intelligence System &nbsp;•&nbsp;
    CatBoost + Cox PH + Groq/Llama &nbsp;•&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
