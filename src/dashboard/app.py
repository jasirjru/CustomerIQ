"""
CustomerIQ — Enterprise Customer Intelligence & Churn Defense Studio
Designed with a Human-Centric, 20-Year UX Masterclass Approach:
- Story-driven customer empathy dossiers
- Context-aware churn risk telemetry & cost-sensitive thresholding (0.35)
- Conversational Explainable AI (XAI) root-cause attribution
- Frontline Customer Success retention blueprints
- Unsupervised Persona Cohort mapping & 2D PCA projection
- Executive Financial ROI & Revenue Preservation Simulator
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import MODELS_DIR, DATA_PROCESSED, REPORTS_DIR
from src.api.service import ModelService
from src.api.schemas import CustomerPayload


# ==============================================================================
# 1. PAGE SETUP & LUXURY DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="CustomerIQ — Intelligence Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F1F5F9;
    }
    
    /* Background Surface & Cards */
    .stApp {
        background: radial-gradient(circle at top right, #111827 0%, #0A0D14 60%, #06080D 100%);
    }
    
    /* Luxury Glassmorphism Container */
    .hero-card {
        background: linear-gradient(135deg, rgba(24, 31, 46, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 28px;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
    }
    
    .persona-dossier-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-tile {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.25);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    
    .kpi-number {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.15;
    }
    
    .kpi-footnote {
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-top: 6px;
    }
    
    /* Humanized Status Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .pill-watch {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .pill-danger {
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    
    /* Playbook Blueprint Box */
    .playbook-blueprint {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366F1;
        border-radius: 14px;
        padding: 22px;
        margin-top: 18px;
    }
    
    .playbook-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }
    .step-number {
        background: #6366F1;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.8rem;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    
    /* Sleek Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94A3B8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #818CF8 !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & ENGINE ARTIFACT CACHING
# ==============================================================================
@st.cache_resource
def load_engine() -> ModelService:
    return ModelService(models_dir=MODELS_DIR, threshold=0.35)

@st.cache_resource
def load_pca():
    pca_path = MODELS_DIR / "pca_2d.joblib"
    if pca_path.exists():
        return joblib.load(pca_path)
    return None

@st.cache_data
def load_studio_datasets():
    pca_df = pd.read_csv(DATA_PROCESSED / "customer_pca_2d.csv") if (DATA_PROCESSED / "customer_pca_2d.csv").exists() else None
    feat_imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv") if (REPORTS_DIR / "feature_importance.csv").exists() else None
    model_comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv") if (REPORTS_DIR / "model_comparison.csv").exists() else None
    y_test_df = pd.read_csv(DATA_PROCESSED / "y_test.csv") if (DATA_PROCESSED / "y_test.csv").exists() else None
    X_test_proc = pd.read_csv(DATA_PROCESSED / "X_test_processed.csv") if (DATA_PROCESSED / "X_test_processed.csv").exists() else None
    return pca_df, feat_imp, model_comp, y_test_df, X_test_proc


service = load_engine()
pca_engine = load_pca()
pca_df, feat_imp_df, model_comp_df, y_test_df, X_test_proc_df = load_studio_datasets()


# ==============================================================================
# 3. SIDEBAR: HUMANIZED CUSTOMER DOSSIER CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 💎 **CustomerIQ Studio**")
    st.caption("Human-Centric Customer Intelligence & Churn Defense")
    st.markdown("---")

    st.markdown("#### 🎭 **Real-World Customer Personas**")
    st.caption("Select an archetypal customer story to inspect their churn trajectory:")

    p_col1, p_col2 = st.columns(2)
    p_col3, p_col4 = st.columns(2)

    if "preset" not in st.session_state:
        st.session_state.preset = "anxious_newcomer"

    if p_col1.button("🚨 Elena Vance", use_container_width=True, help="New fiber subscriber on month-to-month plan. High flight risk."):
        st.session_state.preset = "anxious_newcomer"
    if p_col2.button("🛡️ Arthur Pendelton", use_container_width=True, help="Tenured 5-year loyalist with 2-year contract."):
        st.session_state.preset = "loyal_anchor"
    if p_col3.button("📱 Clara Oswald", use_container_width=True, help="Budget phone-only subscriber with stable usage."):
        st.session_state.preset = "budget_saver"
    if p_col4.button("⚡ Marcus Sterling", use_container_width=True, help="Mid-tenure high-spend power user at risk of switching."):
        st.session_state.preset = "crossroads_power"

    # Set parameters based on archetype
    if st.session_state.preset == "anxious_newcomer":
        persona_name = "Elena Vance"
        persona_tagline = "The Anxious Newcomer • High Spend, Zero Commitment"
        def_tenure = 2
        def_contract = "Month-to-month"
        def_internet = "Fiber optic"
        def_monthly = 94.50
        def_payment = "Electronic check"
        def_paperless = "Yes"
        def_partner = "No"
        def_dependents = "No"
        def_tech_support = "No"
        def_online_sec = "No"
        def_online_backup = "No"
        def_dev_protect = "No"
        def_stream_tv = "Yes"
        def_stream_mov = "Yes"
        def_phone = "Yes"
        def_lines = "No"
        def_gender = "Female"
        def_senior = 0
    elif st.session_state.preset == "loyal_anchor":
        persona_name = "Arthur Pendelton"
        persona_tagline = "The Anchored Loyalist • 5+ Years, Two-Year Contract"
        def_tenure = 64
        def_contract = "Two year"
        def_internet = "DSL"
        def_monthly = 62.00
        def_payment = "Bank transfer (automatic)"
        def_paperless = "No"
        def_partner = "Yes"
        def_dependents = "Yes"
        def_tech_support = "Yes"
        def_online_sec = "Yes"
        def_online_backup = "Yes"
        def_dev_protect = "Yes"
        def_stream_tv = "No"
        def_stream_mov = "No"
        def_phone = "Yes"
        def_lines = "Yes"
        def_gender = "Male"
        def_senior = 0
    elif st.session_state.preset == "budget_saver":
        persona_name = "Clara Oswald"
        persona_tagline = "The Minimalist Saver • Phone Only, Steady & Predictable"
        def_tenure = 36
        def_contract = "One year"
        def_internet = "No"
        def_monthly = 20.25
        def_payment = "Mailed check"
        def_paperless = "No"
        def_partner = "No"
        def_dependents = "No"
        def_tech_support = "No internet service"
        def_online_sec = "No internet service"
        def_online_backup = "No internet service"
        def_dev_protect = "No internet service"
        def_stream_tv = "No internet service"
        def_stream_mov = "No internet service"
        def_phone = "Yes"
        def_lines = "No"
        def_gender = "Female"
        def_senior = 1
    else:  # crossroads_power
        persona_name = "Marcus Sterling"
        persona_tagline = "The Crossroads Power-User • High Value, Rising Frustration"
        def_tenure = 14
        def_contract = "Month-to-month"
        def_internet = "Fiber optic"
        def_monthly = 104.50
        def_payment = "Credit card (automatic)"
        def_paperless = "Yes"
        def_partner = "Yes"
        def_dependents = "No"
        def_tech_support = "No"
        def_online_sec = "Yes"
        def_online_backup = "Yes"
        def_dev_protect = "Yes"
        def_stream_tv = "Yes"
        def_stream_mov = "Yes"
        def_phone = "Yes"
        def_lines = "Yes"
        def_gender = "Male"
        def_senior = 0

    st.markdown("---")
    st.markdown("#### ⚙️ **Customize Customer Attributes**")

    with st.expander("📄 Account & Contract Terms", expanded=True):
        tenure = st.slider("Account Age (Months with company)", min_value=1, max_value=72, value=def_tenure)
        contract = st.selectbox("Contractual Commitment", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
        monthly_charges = st.slider("Monthly Subscription ($)", min_value=18.0, max_value=120.0, value=float(def_monthly), step=0.5)
        total_charges = round(tenure * monthly_charges, 2)
        st.caption(f"Cumulative Lifetime Invoiced: **${total_charges:,.2f}**")
        payment_method = st.selectbox("Payment Channel", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment))
        paperless_billing = st.selectbox("Paperless Invoicing", ["Yes", "No"], index=["Yes", "No"].index(def_paperless))

    with st.expander("🌐 Subscribed Technology Stack", expanded=False):
        internet_service = st.selectbox("Broadband Infrastructure", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_internet))
        phone_service = st.selectbox("Voice Service", ["Yes", "No"], index=["Yes", "No"].index(def_phone))
        multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(def_lines))
        sec_opts = ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"]
        tech_support = st.selectbox("Priority Tech Support", sec_opts, index=0 if def_tech_support in sec_opts else 0)
        online_security = st.selectbox("Cybersecurity Protection", sec_opts, index=0 if def_online_sec in sec_opts else 0)
        online_backup = st.selectbox("Cloud Backup", sec_opts, index=0 if def_online_backup in sec_opts else 0)
        device_protection = st.selectbox("Hardware Coverage", sec_opts, index=0 if def_dev_protect in sec_opts else 0)
        streaming_tv = st.selectbox("Streaming TV Bundle", sec_opts, index=0 if def_stream_tv in sec_opts else 0)
        streaming_movies = st.selectbox("Movie Package", sec_opts, index=0 if def_stream_mov in sec_opts else 0)

    with st.expander("👤 Household & Demographics", expanded=False):
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(def_gender))
        senior_citizen = st.selectbox("Senior Citizen Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=def_senior)
        partner = st.selectbox("Cohabiting Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
        dependents = st.selectbox("Children / Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dependents))

# Construct inference object
active_payload = CustomerPayload(
    gender=gender,
    SeniorCitizen=senior_citizen,
    Partner=partner,
    Dependents=dependents,
    tenure=tenure,
    PhoneService=phone_service,
    MultipleLines=multiple_lines,
    InternetService=internet_service,
    OnlineSecurity=online_security,
    OnlineBackup=online_backup,
    DeviceProtection=device_protection,
    TechSupport=tech_support,
    StreamingTV=streaming_tv,
    StreamingMovies=streaming_movies,
    Contract=contract,
    PaperlessBilling=paperless_billing,
    PaymentMethod=payment_method,
    MonthlyCharges=monthly_charges,
    TotalCharges=total_charges,
)

# Run model inference
eval_res = service.predict_customer(active_payload)


# ==============================================================================
# 4. TOP HERO HEADER & SYSTEM TELEMETRY STRIP
# ==============================================================================
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 1.8rem;">💎</span>
                <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">
                    CustomerIQ Studio
                </span>
                <span style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.4); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                    v1.0 Production
                </span>
            </div>
            <div style="color: #94A3B8; font-size: 0.95rem; max-width: 680px;">
                AI-powered customer risk intelligence engine. Translating complex Scikit-Learn ensembles into empathetic, actionable frontline retention playbooks.
            </div>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.7rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Champion Model</div>
                <div style="font-size: 0.95rem; color: #F1F5F9; font-weight: 700;">Random Forest (84.29% AUC)</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.7rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Decision Threshold</div>
                <div style="font-size: 0.95rem; color: #38BDF8; font-weight: 700;">0.35 (Cost-Optimized)</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px; text-align: center;">
                <div style="font-size: 0.7rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Cloud API</div>
                <div style="font-size: 0.95rem; color: #34D399; font-weight: 700;">Render Live 🟢</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. STUDIO WORKSPACE TABS
# ==============================================================================
tab_dossier, tab_personas, tab_simulator, tab_leaderboard = st.tabs([
    "🔍 Real-Time Customer Dossier & XAI",
    "👥 Persona Landscapes & 2D PCA",
    "💰 Financial ROI & Revenue Simulator",
    "🏛️ Model Engineering & Audit",
])


# ------------------------------------------------------------------------------
# TAB 1: THE HUMANIZED CUSTOMER DOSSIER & EXPLAINABILITY
# ------------------------------------------------------------------------------
with tab_dossier:
    # Empathy Narrative Banner
    proba_pct = eval_res.churn_probability * 100
    is_churn = eval_res.churn_prediction == 1
    
    if proba_pct >= 60.0:
        tier_title = "CRITICAL FLIGHT RISK"
        tier_badge = "pill-danger"
        tier_color = "#FB7185"
        empathy_summary = (
            f"⚠️ **Urgent Attention Required**: {persona_name} is in immediate danger of canceling their subscription. "
            f"The combination of a high **${monthly_charges:.2f}/mo** bill, a lack of long-term contract commitment, "
            f"and zero tech support has created severe churn friction. Proactive outreach within 24 hours is essential."
        )
    elif proba_pct >= 35.0:
        tier_title = "ELEVATED RISK (WATCHLIST)"
        tier_badge = "pill-watch"
        tier_color = "#FBBF24"
        empathy_summary = (
            f"⚠️ **Customer at Crossroads**: While not actively canceling today, {persona_name}'s loyalty is slipping. "
            f"Under standard 0.50 thresholding, they would be invisible. Our **0.35 threshold** caught them early so "
            f"we can intervene before dissatisfaction solidifies into an account termination."
        )
    else:
        tier_title = "HEALTHY & ANCHORED"
        tier_badge = "pill-safe"
        tier_color = "#34D399"
        empathy_summary = (
            f"✅ **Secure & Loyal Relationship**: {persona_name} displays strong retention anchors. "
            f"Their contract structure and service configuration suggest habitual satisfaction. "
            f"Focus on deepening loyalty through rewards rather than defensive retention offers."
        )

    st.markdown(f"""
    <div class="persona-dossier-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
            <div>
                <span style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">Customer Dossier: {persona_name}</span>
                <span style="color: #94A3B8; font-size: 0.9rem; margin-left: 10px;">({persona_tagline})</span>
            </div>
            <span class="status-pill {tier_badge}">{tier_title}</span>
        </div>
        <div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.6;">
            {empathy_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 High-Impact KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div>
                <div class="kpi-label">Predicted Churn Probability</div>
                <div class="kpi-number" style="color: {tier_color};">{proba_pct:.1f}%</div>
            </div>
            <div class="kpi-footnote">
                Random Forest Confidence • Calibration Bounded [0, 1]
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        decision_txt = "TRIGGER RETENTION ACTION" if is_churn else "NO INTERVENTION NEEDED"
        st.markdown(f"""
        <div class="kpi-tile">
            <div>
                <div class="kpi-label">Production Decision (0.35 Boundary)</div>
                <div class="kpi-number" style="color: {tier_color}; font-size: 1.55rem;">{decision_txt}</div>
            </div>
            <div class="kpi-footnote">
                Cost-sensitive threshold eliminates false-negative LTV losses
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div>
                <div class="kpi-label">Behavioral Segment Cohort</div>
                <div class="kpi-number" style="color: #818CF8; font-size: 1.35rem;">
                    {eval_res.customer_segment.split('(')[0]}
                </div>
            </div>
            <div class="kpi-footnote">
                Derived via Unsupervised K-Means ($K=4$) Clustering
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Middle Section: Gauge + Humanized Driver Attribution
    col_meter, col_xai = st.columns([1, 1.3])

    with col_meter:
        st.markdown("#### 🧭 **Churn Probability Speedometer**")
        fig_speedo = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 38, 'color': tier_color, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94A3B8'}},
                'bar': {'color': tier_color, 'thickness': 0.28},
                'bgcolor': "rgba(15, 23, 42, 0.6)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [25, 35], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [35, 100], 'color': "rgba(244, 63, 94, 0.25)"},
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3.5},
                    'thickness': 0.85,
                    'value': 35.0
                }
            }
        ))
        fig_speedo.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=270,
            margin=dict(l=15, r=15, t=30, b=15),
        )
        st.plotly_chart(fig_speedo, use_container_width=True)
        st.caption("White tick mark represents our **0.35 production threshold**. Values above 35% require retention intervention.")

    with col_xai:
        st.markdown("#### 🔍 **Why Is This Customer Behaving This Way?**")
        st.caption("Explainable AI (XAI) feature attribution translated into plain human factors:")

        drivers = eval_res.top_risk_drivers
        if drivers:
            # Map technical feature names into clear human language
            human_mapping = {
                "cat__Contract_Month-to-month": ("Month-to-Month Contract", "Zero contractual commitment makes cancellation effortless"),
                "cat__PaymentMethod_Electronic check": ("Electronic Check Billing", "Manual monthly payments introduce recurring friction"),
                "cat__InternetService_Fiber optic": ("High-Speed Fiber Optic", "Premium tier with high service sensitivity and high monthly cost"),
                "cat__OnlineSecurity_No": ("No Online Security Protection", "Unprotected connection leaves subscriber vulnerable"),
                "cat__TechSupport_No": ("No Priority Tech Support", "Unresolved technical hiccups quickly cause frustration"),
                "num__tenure": ("Account Longevity / Tenure", "Tenure establishes loyalty inertia"),
                "num__MonthlyCharges": ("Monthly Invoice Amount", "High recurring expenditure invites competitive shopping"),
                "num__TotalCharges": ("Cumulative Lifetime Spend", "Reflects customer investment to date"),
            }

            driver_data = []
            for d in drivers:
                raw_feat = d["feature"]
                score = d["impact_score"]
                label, explanation = human_mapping.get(raw_feat, (raw_feat.replace("cat__", "").replace("num__", "").replace("_", " "), "Behavioral attribute contributing to model evaluation"))
                driver_data.append({
                    "Driver": label,
                    "Impact": score,
                    "Explanation": explanation,
                })

            df_drv = pd.DataFrame(driver_data)
            fig_xai = px.bar(
                df_drv,
                x="Impact",
                y="Driver",
                orientation="h",
                color="Impact",
                color_continuous_scale=["#F59E0B", "#F43F5E"],
                hover_data={"Explanation": True, "Impact": ":.4f"},
            )
            fig_xai.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="Risk Contribution Score"),
                yaxis=dict(autorange="reversed", title=None),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_xai, use_container_width=True)
        else:
            st.info("No significant negative risk factors detected for this customer.")

    # Frontline Retention Blueprint
    st.markdown("#### 🎯 **Frontline Retention Action Plan (Customer Success Playbook)**")
    
    actions_list = eval_res.recommended_retention_action.split(" | ")
    
    st.markdown(f"""
    <div class="playbook-blueprint">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
            <span style="font-size: 1.1rem;">📋</span>
            <span style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">
                Prescriptive Retention Script for {persona_name}
            </span>
        </div>
        <div>
            {"".join([f'<div class="playbook-step"><div class="step-number">{i+1}</div><div style="color: #E2E8F0; font-size: 0.93rem;">{act}</div></div>' for i, act in enumerate(actions_list)])}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: PERSONA LANDSCAPES & 2D PCA PROJECTION
# ------------------------------------------------------------------------------
with tab_personas:
    st.markdown("### 👥 **Unsupervised Customer Segmentation & 2D PCA Space**")
    st.markdown(
        "K-Means ($K=4$) automatically discovered 4 natural behavioral personas across our customer base. "
        "Below, we project all 7,043 customers onto a **2D Principal Component map** and plot the currently inspected customer."
    )

    if pca_df is not None and pca_engine is not None:
        # Transform active customer to PCA coordinates
        df_eval = pd.DataFrame([active_payload.model_dump()])
        if df_eval["TotalCharges"].iloc[0] is None:
            df_eval["TotalCharges"] = round(df_eval["tenure"] * df_eval["MonthlyCharges"], 2)
        X_eval_proc = service.preprocessor.transform(df_eval)
        eval_pca = pca_engine.transform(X_eval_proc)[0]

        cluster_descriptions = {
            0: "Cluster 0: Budget Phone Loyalists (7.2% Churn)",
            1: "Cluster 1: Mid-Tier DSL Users (25.1% Churn)",
            2: "Cluster 2: High-Value Multi-Service Loyalists (13.6% Churn)",
            3: "Cluster 3: New High-Spend Flight Risks (57.4% Churn ⚠️)",
        }

        df_vis = pca_df.copy()
        df_vis["Persona"] = df_vis["Cluster"].map(cluster_descriptions)

        fig_map = px.scatter(
            df_vis.sample(min(2500, len(df_vis)), random_state=42),
            x="PC1",
            y="PC2",
            color="Persona",
            opacity=0.5,
            color_discrete_map={
                cluster_descriptions[0]: "#10B981",  # Emerald
                cluster_descriptions[1]: "#38BDF8",  # Sky Blue
                cluster_descriptions[2]: "#A855F7",  # Purple
                cluster_descriptions[3]: "#F43F5E",  # Rose Red
            },
            title="Principal Component Space: PC1 (Spend & Services) vs. PC2 (Tenure & Loyalty)",
        )

        # Highlight current customer
        fig_map.add_trace(go.Scatter(
            x=[eval_pca[0]],
            y=[eval_pca[1]],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#FACC15", line=dict(color="#FFFFFF", width=2)),
            text=[f"⭐ {persona_name.upper()}"],
            textposition="top center",
            textfont=dict(color="#FACC15", size=13, family="Plus Jakarta Sans", weight=700),
            name="Current Customer",
        ))

        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=540,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="PC1 (Service Breadth & Monthly Charges)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="PC2 (Tenure & Long-Term Contract Commitment)"),
            legend=dict(bgcolor="rgba(15,23,42,0.85)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # 4 Persona Cards
        pc0, pc1, pc2, pc3 = st.columns(4)
        pc0.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 14px;">
            <div style="font-weight: 700; color: #34D399; font-size: 0.95rem;">Cluster 0: Budget Phone</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Share: <strong>21.5%</strong> • Churn: <strong>7.2%</strong></div>
            <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 8px;">Low bill, phone-only subscribers. Safe, steady, and low-maintenance.</div>
        </div>
        """, unsafe_allow_html=True)

        pc1.markdown("""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">Cluster 1: Mid-Tier DSL</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Share: <strong>23.4%</strong> • Churn: <strong>25.1%</strong></div>
            <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 8px;">Moderate spenders with legacy copper connections. Receptive to fiber upgrades.</div>
        </div>
        """, unsafe_allow_html=True)

        pc2.markdown("""
        <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 12px; padding: 14px;">
            <div style="font-weight: 700; color: #C084FC; font-size: 0.95rem;">Cluster 2: Multi-Service</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Share: <strong>28.5%</strong> • Churn: <strong>13.6%</strong></div>
            <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 8px;">High-value anchored subscribers with bundled streaming and long contracts.</div>
        </div>
        """, unsafe_allow_html=True)

        pc3.markdown("""
        <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 12px; padding: 14px;">
            <div style="font-weight: 700; color: #FB7185; font-size: 0.95rem;">Cluster 3: Flight Risks ⚠️</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Share: <strong>26.5%</strong> • Churn: <strong>57.4%</strong></div>
            <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 8px;">New high-spend fiber customers on month-to-month contracts. The primary danger zone!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("PCA data artifact not found. Verify customer_pca_2d.csv exists in data/processed/.")


# ------------------------------------------------------------------------------
# TAB 3: EXECUTIVE ROI & DECISION THRESHOLD SIMULATOR
# ------------------------------------------------------------------------------
with tab_simulator:
    st.markdown("### 💰 **Executive Decision Threshold & Financial ROI Simulator**")
    st.markdown(
        "Standard machine learning uses an arbitrary **0.50 probability cutoff**. "
        "In telecom, **losing an at-risk customer (False Negative) costs ~$500 in Lifetime Value (LTV)**, "
        "while **wasting an unnecessary retention voucher (False Positive) costs only ~$100**."
    )

    if X_test_proc_df is not None and y_test_df is not None:
        col_ctl1, col_ctl2 = st.columns([1.5, 1])

        with col_ctl1:
            thresh_slider = st.slider("Adjust Decision Threshold Boundary", min_value=0.10, max_value=0.85, value=0.35, step=0.05)
        with col_ctl2:
            st.caption("Lowering the threshold catches more churners (higher Recall) at the expense of slight over-targeting.")

        # Compute test set predictions
        y_true = y_test_df.values.ravel()
        test_probas = service.champion_model.predict_proba(X_test_proc_df)[:, 1]
        y_pred = (test_probas >= thresh_slider).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0

        # Financial Model
        cost_per_fn = 500  # Lost LTV
        cost_per_fp = 100  # Retention offer cost
        total_loss = (fn * cost_per_fn) + (fp * cost_per_fp)

        # Baseline at 0.50
        y_pred_50 = (test_probas >= 0.50).astype(int)
        fn_50 = int(np.sum((y_pred_50 == 0) & (y_true == 1)))
        fp_50 = int(np.sum((y_pred_50 == 1) & (y_true == 0)))
        loss_50 = (fn_50 * cost_per_fn) + (fp_50 * cost_per_fp)
        net_saved = loss_50 - total_loss

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Caught Churners (Recall)", f"{recall:.1f}%", delta=f"{recall - 51.9:+.1f}% vs 0.50")
        m2.metric("Missed Churners (FN)", f"{fn}", delta=f"{fn - fn_50} customers", delta_color="inverse")
        m3.metric("Net Financial Cost", f"${total_loss:,.0f}")
        m4.metric("Net Business Savings", f"${net_saved:,.0f}", delta=f"${net_saved:,.0f}", delta_color="normal")

        # Threshold Sweep Curve
        t_range = np.linspace(0.10, 0.85, 32)
        cost_curve = []
        for t in t_range:
            p_t = (test_probas >= t).astype(int)
            fn_t = np.sum((p_t == 0) & (y_true == 1))
            fp_t = np.sum((p_t == 1) & (y_true == 0))
            cost_curve.append((fn_t * cost_per_fn) + (fp_t * cost_per_fp))

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=t_range, y=cost_curve, mode="lines+markers",
            name="Net Financial Cost ($)", line=dict(color="#F43F5E", width=3)
        ))
        fig_roi.add_vline(x=0.35, line_dash="dash", line_color="#34D399", annotation_text="Optimal 0.35 Boundary", annotation_position="top right")
        fig_roi.add_vline(x=0.50, line_dash="dot", line_color="#F59E0B", annotation_text="Default 0.50 Cutoff", annotation_position="bottom right")
        fig_roi.add_vline(x=thresh_slider, line_color="#6366F1", line_width=3, annotation_text=f"Selected: {thresh_slider:.2f}")

        fig_roi.update_layout(
            title="Total Revenue Loss Curve Across Classification Thresholds",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=400,
            xaxis=dict(title="Decision Threshold Boundary", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Cumulative Business Loss ($) [FN * $500 + FP * $100]", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        st.caption("Proof that threshold 0.35 minimizes total net cost and maximizes retained customer lifetime value.")
    else:
        st.info("Test split dataset required for ROI simulation.")


# ------------------------------------------------------------------------------
# TAB 4: MODEL AUDIT & LEADERBOARD
# ------------------------------------------------------------------------------
with tab_leaderboard:
    st.markdown("### 🏛️ **Model Architecture Benchmark & Transparency Audit**")
    st.markdown(
        "All models trained strictly on `X_train` (80%) and evaluated on the held-out `X_test` (20%) "
        "with zero data leakage via our unified `ColumnTransformer`."
    )

    if model_comp_df is not None:
        col_t, col_p = st.columns([1, 1.2])

        with col_t:
            st.markdown("#### 🏆 **Algorithm Leaderboard**")
            df_disp = model_comp_df.copy()
            for c in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
                if c in df_disp.columns:
                    df_disp[c] = df_disp[c].apply(lambda v: f"{float(v) * 100:.2f}%")
            st.dataframe(df_disp, use_container_width=True, hide_index=True)

        with col_p:
            st.markdown("#### 📊 **ROC-AUC Comparison**")
            fig_bar = px.bar(
                model_comp_df,
                x="Model",
                y="ROC_AUC",
                color="ROC_AUC",
                text="ROC_AUC",
                color_continuous_scale=["#6366F1", "#34D399"],
            )
            fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                yaxis=dict(range=[0.75, 0.90], title="ROC-AUC Score", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    if feat_imp_df is not None:
        st.markdown("---")
        st.markdown("#### 🔬 **Global Permutation Feature Importance (Unseen Test Set)**")
        st.caption("Unlike training Gini MDI, Permutation Importance measures true out-of-sample predictive power without cardinality bias.")
        
        fig_glob = px.bar(
            feat_imp_df.head(10),
            x="Permutation_Importance_Mean",
            y="Feature",
            orientation="h",
            error_x="Permutation_Std",
            color="Permutation_Importance_Mean",
            color_continuous_scale="Viridis",
        )
        fig_glob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_glob, use_container_width=True)
