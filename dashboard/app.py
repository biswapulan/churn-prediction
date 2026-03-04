# dashboard/app.py
# Streamlit Dashboard for Customer Churn Prediction

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Churn Predictor",
    page_icon  = "📊",
    layout     = "wide"
)

# ─────────────────────────────────────────
# API URL
# ─────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def get_risk_color(risk_level):
    if "High" in risk_level:
        return "#e74c3c"
    elif "Medium" in risk_level:
        return "#f39c12"
    else:
        return "#2ecc71"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("📊 Churn Predictor")
st.sidebar.markdown("---")

# API Health Status
api_healthy = check_api_health()
if api_healthy:
    st.sidebar.success("🟢 API Connected")
else:
    st.sidebar.error("🔴 API Offline — Start FastAPI first!")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "👤 Single Prediction", "📂 Batch Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("XGBoost • SHAP • FastAPI • Streamlit")

# ─────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("### Predict which customers are at risk of leaving")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("👤 **Single Prediction**\n\nEnter one customer's details and get instant churn prediction with explanation")

    with col2:
        st.info("📂 **Batch Prediction**\n\nUpload a CSV file of customers and get predictions for all of them at once")

    with col3:
        st.info("🔍 **SHAP Explanations**\n\nUnderstand WHY the model predicts churn for each customer")

    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Make sure the **API is connected** (green indicator in sidebar)
    2. Go to **Single Prediction** to predict one customer
    3. Go to **Batch Prediction** to upload a CSV and predict many customers
    """)

# ─────────────────────────────────────────
# PAGE 2 — SINGLE PREDICTION
# ─────────────────────────────────────────
elif page == "👤 Single Prediction":
    st.title("👤 Single Customer Prediction")
    st.markdown("Fill in the customer details below and click **Predict**")
    st.markdown("---")

    # Form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen  = st.selectbox("Senior Citizen", [0, 1])
        partner         = st.selectbox("Partner", ["Yes", "No"])
        dependents      = st.selectbox("Dependents", ["Yes", "No"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines",
                            ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service",
                            ["Fiber optic", "DSL", "No"])
        online_security  = st.selectbox("Online Security",
                            ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup",
                            ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection",
                            ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support",
                            ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",
                            ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",
                            ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Billing")
        contract          = st.selectbox("Contract",
                            ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method    = st.selectbox("Payment Method", [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"
                            ])
        monthly_charges   = st.number_input("Monthly Charges ($)",
                            min_value=0.0, max_value=200.0, value=70.0)
        total_charges     = st.number_input("Total Charges ($)",
                            min_value=0.0, max_value=10000.0,
                            value=monthly_charges * tenure)

    st.markdown("---")

    # Predict Button
    if st.button("🔍 Predict Churn", use_container_width=True):
        if not api_healthy:
            st.error("❌ API is offline! Start FastAPI first.")
        else:
            with st.spinner("Analyzing customer..."):
                customer_data = {
                    "gender"           : gender,
                    "SeniorCitizen"    : senior_citizen,
                    "Partner"          : partner,
                    "Dependents"       : dependents,
                    "tenure"           : tenure,
                    "PhoneService"     : phone_service,
                    "MultipleLines"    : multiple_lines,
                    "InternetService"  : internet_service,
                    "OnlineSecurity"   : online_security,
                    "OnlineBackup"     : online_backup,
                    "DeviceProtection" : device_protection,
                    "TechSupport"      : tech_support,
                    "StreamingTV"      : streaming_tv,
                    "StreamingMovies"  : streaming_movies,
                    "Contract"         : contract,
                    "PaperlessBilling" : paperless_billing,
                    "PaymentMethod"    : payment_method,
                    "MonthlyCharges"   : monthly_charges,
                    "TotalCharges"     : total_charges
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer_data
                )
                result = response.json()

            # ── Results ──
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                prediction_text = "⚠️ WILL CHURN" if result['churn_prediction'] == 1 else "✅ WILL STAY"
                color = "#e74c3c" if result['churn_prediction'] == 1 else "#2ecc71"
                st.markdown(f"""
                <div style='background-color:{color};padding:20px;
                border-radius:10px;text-align:center;color:white;
                font-size:20px;font-weight:bold'>
                {prediction_text}
                </div>""", unsafe_allow_html=True)

            with col2:
                prob = result['churn_probability'] * 100
                fig  = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = prob,
                    title = {"text": "Churn Probability %"},
                    gauge = {
                        "axis"  : {"range": [0, 100]},
                        "bar"   : {"color": get_risk_color(result['risk_level'])},
                        "steps" : [
                            {"range": [0, 40],  "color": "#d5f5e3"},
                            {"range": [40, 70], "color": "#fdebd0"},
                            {"range": [70, 100],"color": "#fadbd8"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown(f"**Risk Level:** {result['risk_level']}")
                st.markdown(f"**Message:** {result['message']}")

            # ── SHAP Reasons ──
            st.markdown("---")
            st.subheader("🔍 Why This Prediction?")

            reasons_df = pd.DataFrame(result['top_reasons'])
            reasons_df['color'] = reasons_df['impact'].apply(
                lambda x: '#e74c3c' if x > 0 else '#2ecc71'
            )

            fig = px.bar(
                reasons_df,
                x     = 'impact',
                y     = 'feature',
                color = 'effect',
                orientation = 'h',
                title = 'Top Factors Influencing This Prediction',
                color_discrete_map={
                    'increases churn risk' : '#e74c3c',
                    'decreases churn risk' : '#2ecc71'
                }
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — BATCH PREDICTION
# ─────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Customer Prediction")
    st.markdown("Upload a CSV file to predict churn for multiple customers")
    st.markdown("---")

    # Sample CSV download
    st.markdown("### Step 1 — Download Sample CSV")
    sample_data = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50, "TotalCharges": 171.00
    }])

    st.download_button(
        label     = "📥 Download Sample CSV",
        data      = sample_data.to_csv(index=False),
        file_name = "sample_customer.csv",
        mime      = "text/csv"
    )

    st.markdown("---")
    st.markdown("### Step 2 — Upload Your CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df)} customers**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict All", use_container_width=True):
            if not api_healthy:
                st.error("❌ API is offline!")
            else:
                with st.spinner(f"Predicting churn for {len(df)} customers..."):
                    customers = df.to_dict(orient='records')
                    response  = requests.post(
                        f"{API_URL}/predict/batch",
                        json=customers
                    )
                    result = response.json()

                # ── Summary Metrics ──
                st.markdown("---")
                st.subheader("📊 Batch Results Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers",
                              result['total_customers'])
                with col2:
                    st.metric("High Risk (Churn)",
                              result['high_risk_count'],
                              delta=f"{result['high_risk_count']/result['total_customers']*100:.1f}%")
                with col3:
                    stay_count = result['total_customers'] - result['high_risk_count']
                    st.metric("Low Risk (Stay)", stay_count)

                # ── Results Table ──
                predictions_df = pd.DataFrame(result['predictions'])
                predictions_df = pd.concat(
                    [df.reset_index(drop=True), predictions_df], axis=1
                )

                st.markdown("---")
                st.subheader("📋 Detailed Results")
                st.dataframe(predictions_df, use_container_width=True)

                # ── Risk Distribution Chart ──
                risk_counts = predictions_df['risk_level'].value_counts()
                fig = px.pie(
                    values = risk_counts.values,
                    names  = risk_counts.index,
                    title  = "Risk Level Distribution",
                    color_discrete_map={
                        '🔴 High Risk'   : '#e74c3c',
                        '🟡 Medium Risk' : '#f39c12',
                        '🟢 Low Risk'    : '#2ecc71'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Download Results ──
                st.download_button(
                    label     = "📥 Download Results CSV",
                    data      = predictions_df.to_csv(index=False),
                    file_name = "churn_predictions.csv",
                    mime      = "text/csv"
                )