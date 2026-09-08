"""
CustomerIQ — Interactive Streamlit Customer Intelligence Dashboard
Production-grade dashboard providing:
1. Real-time Churn Prediction with Cost-Sensitive Thresholding (0.35)
2. Explainable AI (XAI) Local Feature Driver Attribution & Retention Playbook
3. Unsupervised Customer Segmentation (K-Means) & 2D PCA Live Mapping
4. Executive Decision Threshold & Financial ROI Simulator
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
# 1. PAGE CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="CustomerIQ — Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Typography & Background Elements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Dark Glassmorphism Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .metric-subtitle {
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-top: 6px;
    }
    
    /* Custom Risk Badges */
    .badge-low {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-mod {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .playbook-box {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. ARTIFACT & RESOURCE CACHING
# ==============================================================================
@st.cache_resource
def load_model_service() -> ModelService:
    return ModelService(models_dir=MODELS_DIR, threshold=0.35)

@st.cache_resource
def load_pca_model():
    pca_path = MODELS_DIR / "pca_2d.joblib"
    if pca_path.exists():
        return joblib.load(pca_path)
    return None

@st.cache_data
def load_precomputed_data():
    pca_df = pd.read_csv(DATA_PROCESSED / "customer_pca_2d.csv") if (DATA_PROCESSED / "customer_pca_2d.csv").exists() else None
    feat_imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv") if (REPORTS_DIR / "feature_importance.csv").exists() else None
    model_comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv") if (REPORTS_DIR / "model_comparison.csv").exists() else None
    y_test_df = pd.read_csv(DATA_PROCESSED / "y_test.csv") if (DATA_PROCESSED / "y_test.csv").exists() else None
    X_test_proc = pd.read_csv(DATA_PROCESSED / "X_test_processed.csv") if (DATA_PROCESSED / "X_test_processed.csv").exists() else None
    return pca_df, feat_imp, model_comp, y_test_df, X_test_proc


service = load_model_service()
pca_transformer = load_pca_model()
pca_df, feat_imp_df, model_comp_df, y_test_df, X_test_proc_df = load_precomputed_data()


# ==============================================================================
# 3. SIDEBAR: CUSTOMER PROFILE CONTROLS & PRESETS
# ==============================================================================
st.sidebar.markdown("## 🧠 **CustomerIQ**")
st.sidebar.caption("Machine Learning & Customer Intelligence Engine")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚡ **Quick Scenario Presets**")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)

# Preset states
if "preset" not in st.session_state:
    st.session_state.preset = "high_risk"

if col_p1.button("🚨 Flight Risk", use_container_width=True, help="New customer, month-to-month, fiber, electronic check"):
    st.session_state.preset = "high_risk"
if col_p2.button("🛡️ Loyalist", use_container_width=True, help="Tenured customer, 2-year contract, DSL, auto-pay"):
    st.session_state.preset = "loyalist"
if col_p3.button("📱 Budget", use_container_width=True, help="Phone only, 1-year contract, low monthly bill"):
    st.session_state.preset = "budget"

# Set defaults based on preset
if st.session_state.preset == "high_risk":
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
    def_gender = "Male"
    def_senior = 0
elif st.session_state.preset == "loyalist":
    def_tenure = 62
    def_contract = "Two year"
    def_internet = "DSL"
    def_monthly = 64.00
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
    def_gender = "Female"
    def_senior = 0
else:  # Budget
    def_tenure = 38
    def_contract = "One year"
    def_internet = "No"
    def_monthly = 20.50
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

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 **Customer Attributes**")

with st.sidebar.expander("💳 Account & Billing", expanded=True):
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=def_tenure)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(def_monthly), step=0.5)
    total_charges = round(tenure * monthly_charges, 2)
    st.caption(f"Estimated Total Charges: **${total_charges:,.2f}**")
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment))
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(def_paperless))

with st.sidebar.expander("🌐 Services Subscribed", expanded=False):
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_internet))
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(def_phone))
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(def_lines))
    
    sec_opts = ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"]
    online_security = st.selectbox("Online Security", sec_opts, index=0 if def_online_sec in sec_opts else 0)
    tech_support = st.selectbox("Tech Support", sec_opts, index=0 if def_tech_support in sec_opts else 0)
    online_backup = st.selectbox("Online Backup", sec_opts, index=0 if def_online_backup in sec_opts else 0)
    device_protection = st.selectbox("Device Protection", sec_opts, index=0 if def_dev_protect in sec_opts else 0)
    streaming_tv = st.selectbox("Streaming TV", sec_opts, index=0 if def_stream_tv in sec_opts else 0)
    streaming_movies = st.selectbox("Streaming Movies", sec_opts, index=0 if def_stream_mov in sec_opts else 0)

with st.sidebar.expander("👤 Customer Demographics", expanded=False):
    gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(def_gender))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=def_senior)
    partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dependents))

# Build CustomerPayload
current_customer_payload = CustomerPayload(
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

# Execute inference via core service
prediction = service.predict_customer(current_customer_payload)


# ==============================================================================
# 4. MAIN APP HEADER & TOP STATUS BANNER
# ==============================================================================
st.markdown("# 🧠 **CustomerIQ — Customer Intelligence Platform**")
st.markdown(
    "Enterprise-grade Machine Learning system powering **Supervised Churn Risk Detection**, "
    "**Explainable AI (XAI)**, and **Unsupervised Persona Segmentation**."
)

st.markdown("""
<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
    <span class="badge-low" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.4);">
        🏆 Champion: Random Forest Classifier (ROC-AUC: 84.29%)
    </span>
    <span class="badge-low" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(168, 85, 247, 0.4);">
        ⚖️ Production Threshold: 0.35 (Recall: 70.6%)
    </span>
    <span class="badge-low" style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border-color: rgba(34, 197, 94, 0.4);">
        🔒 Zero Test Leakage Preprocessor
    </span>
    <span class="badge-low" style="background: rgba(234, 179, 8, 0.15); color: #facc15; border-color: rgba(234, 179, 8, 0.4);">
        🚀 Live Microservice on Render
    </span>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. TAB NAVIGATION
# ==============================================================================
tab_prediction, tab_segmentation, tab_roi, tab_models = st.tabs([
    "🔮 Real-Time Risk & XAI Playbook",
    "👥 Unsupervised Personas & 2D PCA",
    "💰 Executive Threshold & ROI Simulator",
    "🏛️ Model Benchmark & Leaderboard",
])


# ------------------------------------------------------------------------------
# TAB 1: REAL-TIME CHURN RISK & EXPLAINABILITY
# ------------------------------------------------------------------------------
with tab_prediction:
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        risk_color = "#f87171" if prediction.risk_level == "HIGH" else ("#fbbf24" if prediction.risk_level == "MODERATE" else "#34d399")
        badge_cls = "badge-high" if prediction.risk_level == "HIGH" else ("badge-mod" if prediction.risk_level == "MODERATE" else "badge-low")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Churn Probability</div>
            <div class="metric-value" style="color: {risk_color};">{prediction.churn_probability * 100:.1f}%</div>
            <div class="metric-subtitle">
                Risk Tier: <span class="{badge_cls}">{prediction.risk_level} RISK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        outcome_label = "CHURN FLIGHT RISK" if prediction.churn_prediction == 1 else "RETAINED / HEALTHY"
        outcome_color = "#f87171" if prediction.churn_prediction == 1 else "#34d399"
        sub_desc = "Exceeds 0.35 threshold (intervention needed)" if prediction.churn_prediction == 1 else "Below 0.35 threshold"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Decision (Threshold = 0.35)</div>
            <div class="metric-value" style="color: {outcome_color}; font-size: 1.6rem;">{outcome_label}</div>
            <div class="metric-subtitle">{sub_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Assigned Customer Segment</div>
            <div class="metric-value" style="color: #60a5fa; font-size: 1.35rem;">{prediction.customer_segment.split('(')[0]}</div>
            <div class="metric-subtitle">Persona: {prediction.customer_segment}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_gauge, col_drivers = st.columns([1, 1.2])

    with col_gauge:
        st.markdown("#### 🧭 **Churn Probability Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction.churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#cbd5e1'}},
            number={'suffix': "%", 'font': {'size': 36, 'color': risk_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': risk_color, 'thickness': 0.3},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.25)"},
                    {'range': [25, 35], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [35, 100], 'color': "rgba(239, 68, 68, 0.25)"},
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 4},
                    'thickness': 0.8,
                    'value': 35.0
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("White marker indicates production decision threshold (35%). Customers above 35% trigger retention workflows.")

    with col_drivers:
        st.markdown("#### 🔍 **Local Risk Attribution (Explainable AI)**")
        # Extract features and impacts
        drivers = prediction.top_risk_drivers
        if drivers:
            df_drv = pd.DataFrame(drivers)
            # Friendly name translation
            clean_names = []
            for name in df_drv["feature"]:
                clean = name.replace("cat__", "").replace("num__", "").replace("_", ": ")
                clean_names.append(clean)
            df_drv["feature_clean"] = clean_names

            fig_drv = px.bar(
                df_drv,
                x="impact_score",
                y="feature_clean",
                orientation="h",
                color="impact_score",
                color_continuous_scale=["#f59e0b", "#ef4444"],
                labels={"impact_score": "Risk Contribution Score", "feature_clean": "Attribute"},
            )
            fig_drv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_drv, use_container_width=True)
        else:
            st.info("No significant negative risk factors identified for this profile.")

    # Retention Playbook Card
    st.markdown("#### 🎯 **Prescriptive Retention Playbook**")
    st.markdown(f"""
    <div class="playbook-box">
        <div style="font-weight: 700; color: #93c5fd; font-size: 1.05rem; margin-bottom: 6px;">
            Recommended Action Plan:
        </div>
        <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.5;">
            {prediction.recommended_retention_action}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: UNSUPERVISED PERSONAS & 2D PCA PROJECTION
# ------------------------------------------------------------------------------
with tab_segmentation:
    st.markdown("### 👥 **Unsupervised Customer Segmentation & 2D PCA Space**")
    st.markdown(
        "K-Means ($K=4$) discovered distinct behavioral cohorts. We project the full customer base onto "
        "the first 2 Principal Components (explaining 27.5% total variance) and map the currently evaluated customer."
    )

    if pca_df is not None and pca_transformer is not None:
        # Project current customer to 2D PCA space
        df_single = pd.DataFrame([current_customer_payload.model_dump()])
        if df_single["TotalCharges"].iloc[0] is None:
            df_single["TotalCharges"] = round(df_single["tenure"] * df_single["MonthlyCharges"], 2)
        X_cust_arr = service.preprocessor.transform(df_single)
        cust_pc = pca_transformer.transform(X_cust_arr)[0]
        cust_pc1, cust_pc2 = cust_pc[0], cust_pc[1]

        # Map cluster names
        cluster_labels = {
            0: "Cluster 0: Budget Phone Loyalists (7.2% Churn)",
            1: "Cluster 1: Mid-Tier DSL Users (25.1% Churn)",
            2: "Cluster 2: High-Value Multi-Service Loyalists (13.6% Churn)",
            3: "Cluster 3: New High-Spend Flight Risks (57.4% Churn ⚠️)",
        }
        df_plot = pca_df.copy()
        df_plot["Persona"] = df_plot["Cluster"].map(cluster_labels)

        # Plotly PCA Scatter
        fig_pca = px.scatter(
            df_plot.sample(min(2000, len(df_plot)), random_state=42),
            x="PC1",
            y="PC2",
            color="Persona",
            opacity=0.55,
            color_discrete_map={
                cluster_labels[0]: "#10b981",  # Emerald
                cluster_labels[1]: "#3b82f6",  # Blue
                cluster_labels[2]: "#a855f7",  # Purple
                cluster_labels[3]: "#ef4444",  # Crimson
            },
            title="Customer Base Projected on Principal Components (PC1: Spend & Services vs PC2: Tenure & Loyalty)",
        )

        # Add the evaluated customer as a prominent glowing gold star
        fig_pca.add_trace(go.Scatter(
            x=[cust_pc1],
            y=[cust_pc2],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color="#facc15", line=dict(color="#ffffff", width=2)),
            text=["⭐ EVALUATED CUSTOMER"],
            textposition="top center",
            textfont=dict(color="#facc15", size=13, family="Inter"),
            name="Current Customer",
        ))

        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=540,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="PC1 (Service Complexity & Monthly Spend)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="PC2 (Tenure & Long-Term Contract Commitment)"),
            legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        )

        st.plotly_chart(fig_pca, use_container_width=True)

        col_c0, col_c1, col_c2, col_c3 = st.columns(4)
        col_c0.markdown("""
        **Cluster 0: Budget Phone**
        - Share: **21.5%**
        - Churn Rate: **7.2%**
        - Strategy: Autopay incentives
        """)
        col_c1.markdown("""
        **Cluster 1: Mid-Tier DSL**
        - Share: **23.4%**
        - Churn Rate: **25.1%**
        - Strategy: Speed upgrade deals
        """)
        col_c2.markdown("""
        **Cluster 2: Multi-Service**
        - Share: **28.5%**
        - Churn Rate: **13.6%**
        - Strategy: VIP customer loyalty
        """)
        col_c3.markdown("""
        **Cluster 3: Flight Risks**
        - Share: **26.5%**
        - Churn Rate: **57.4% ⚠️**
        - Strategy: Urgent contract discount
        """)
    else:
        st.warning("PCA data artifact not found. Please ensure customer_pca_2d.csv is generated.")


# ------------------------------------------------------------------------------
# TAB 3: EXECUTIVE ROI & DECISION THRESHOLD SIMULATOR
# ------------------------------------------------------------------------------
with tab_roi:
    st.markdown("### 💰 **Cost-Sensitive Decision Threshold & ROI Simulator**")
    st.markdown(
        "Standard classification uses a default 0.50 threshold. In churn management, **a False Negative (losing a customer) "
        "is 5x more costly than a False Positive (wasting a retention discount)**. Slide the threshold below to observe net financial impact."
    )

    if X_test_proc_df is not None and y_test_df is not None:
        sim_threshold = st.slider("Select Decision Threshold", min_value=0.10, max_value=0.85, value=0.35, step=0.05)

        # Predict test probabilities
        y_test = y_test_df.values.ravel()
        probas_test = service.champion_model.predict_proba(X_test_proc_df)[:, 1]
        y_pred = (probas_test >= sim_threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_test == 1)))
        fp = int(np.sum((y_pred == 1) & (y_test == 0)))
        tn = int(np.sum((y_pred == 0) & (y_test == 0)))
        fn = int(np.sum((y_pred == 0) & (y_test == 1)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Cost Modeling Assumptions
        cost_fn = 500  # Customer LTV loss
        cost_fp = 100  # Marketing retention voucher cost
        total_loss = (fn * cost_fn) + (fp * cost_fp)

        # Baseline at 0.50
        y_pred_50 = (probas_test >= 0.50).astype(int)
        fn_50 = int(np.sum((y_pred_50 == 0) & (y_test == 1)))
        fp_50 = int(np.sum((y_pred_50 == 1) & (y_test == 0)))
        loss_50 = (fn_50 * cost_fn) + (fp_50 * cost_fp)
        net_savings = loss_50 - total_loss

        col_roi1, col_roi2, col_roi3, col_roi4 = st.columns(4)
        col_roi1.metric("Recall (Caught Churners)", f"{recall * 100:.1f}%", delta=f"{(recall - (tp / (tp + fn_50))) * 100:+.1f}% vs 0.50")
        col_roi2.metric("Missed Churners (FN)", f"{fn}", delta=f"{fn - fn_50} customers", delta_color="inverse")
        col_roi3.metric("Net Financial Cost", f"${total_loss:,.0f}")
        col_roi4.metric("Savings vs 0.50 Default", f"${net_savings:,.0f}", delta=f"${net_savings:,.0f}", delta_color="normal")

        # Threshold Sweep Curve
        thresholds = np.linspace(0.1, 0.85, 30)
        costs = []
        recalls = []
        for t in thresholds:
            preds_t = (probas_test >= t).astype(int)
            fn_t = np.sum((preds_t == 0) & (y_test == 1))
            fp_t = np.sum((preds_t == 1) & (y_test == 0))
            costs.append((fn_t * cost_fn) + (fp_t * cost_fp))
            recalls.append(np.sum((preds_t == 1) & (y_test == 1)) / (tp + fn))

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=thresholds, y=costs, mode="lines+markers",
            name="Net Financial Cost ($)", line=dict(color="#ef4444", width=3)
        ))
        fig_cost.add_vline(x=0.35, line_dash="dash", line_color="#34d399", annotation_text="Optimal Threshold (0.35)", annotation_position="top right")
        fig_cost.add_vline(x=0.50, line_dash="dot", line_color="#f59e0b", annotation_text="Default Threshold (0.50)", annotation_position="bottom right")
        fig_cost.add_vline(x=sim_threshold, line_color="#3b82f6", line_width=3, annotation_text=f"Current: {sim_threshold:.2f}")

        fig_cost.update_layout(
            title="Total Business Cost Curve Across Classification Thresholds",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=400,
            xaxis=dict(title="Decision Threshold", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Net Cost ($) [FN * $500 + FP * $100]", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("Test split artifacts required for simulation.")


# ------------------------------------------------------------------------------
# TAB 4: MODEL LEADERBOARD & ARCHITECTURE
# ------------------------------------------------------------------------------
with tab_models:
    st.markdown("### 🏛️ **Model Comparison Benchmark (5 Classical Algorithms)**")
    st.markdown(
        "All models evaluated on the held-out 20% stratified test set (`X_test`, `y_test`) under identical preprocessor transformations."
    )

    if model_comp_df is not None:
        # Format metrics table
        df_show = model_comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda x: f"{float(x) * 100:.2f}%")

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Bar chart comparison of ROC-AUC
        fig_bench = px.bar(
            model_comp_df,
            x="Model",
            y="ROC_AUC",
            color="ROC_AUC",
            text="ROC_AUC",
            title="Model Benchmark — ROC-AUC Comparison",
            color_continuous_scale=["#3b82f6", "#10b981"],
        )
        fig_bench.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bench.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            yaxis=dict(range=[0.75, 0.90], title="ROC-AUC Score", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bench, use_container_width=True)
    
    if feat_imp_df is not None:
        st.markdown("#### 🔬 **Global Feature Importance: MDI vs. Permutation Importance**")
        top_feats = feat_imp_df.head(10)
        fig_fi = px.bar(
            top_feats,
            x="Permutation_Importance_Mean",
            y="Feature",
            orientation="h",
            error_x="Permutation_Std",
            title="Top 10 Drivers by Permutation Importance (on held-out test split)",
            color="Permutation_Importance_Mean",
            color_continuous_scale="Viridis",
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
