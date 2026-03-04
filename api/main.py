# api/main.py
# FastAPI Backend for Customer Churn Prediction

import joblib
import numpy as np
import pandas as pd
import shap

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ─────────────────────────────────────────
# LOAD MODEL & PREPROCESSOR AT STARTUP
# ─────────────────────────────────────────
print("Loading model and preprocessor...")

model          = joblib.load(MODELS_DIR / 'best_model.pkl')
preprocessor   = joblib.load(MODELS_DIR / 'preprocessor.pkl')
numerical_cols = joblib.load(MODELS_DIR / 'numerical_cols.pkl')
categorical_cols = joblib.load(MODELS_DIR / 'categorical_cols.pkl')

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Get feature names
ohe_feature_names = preprocessor\
    .named_transformers_['cat']['encoder']\
    .get_feature_names_out(categorical_cols)\
    .tolist()
all_feature_names = numerical_cols + ohe_feature_names

print("Model loaded successfully ✅")

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn Prediction API",
    description = "Predicts customer churn probability with SHAP explanations",
    version     = "1.0.0"
)

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────
# INPUT SCHEMA
# Defines exactly what data the API expects
# Pydantic automatically validates the input
# ─────────────────────────────────────────
class CustomerData(BaseModel):
    gender             : str   = Field(..., example="Male")
    SeniorCitizen      : int   = Field(..., example=0)
    Partner            : str   = Field(..., example="Yes")
    Dependents         : str   = Field(..., example="No")
    tenure             : int   = Field(..., example=12)
    PhoneService       : str   = Field(..., example="Yes")
    MultipleLines      : str   = Field(..., example="No")
    InternetService    : str   = Field(..., example="Fiber optic")
    OnlineSecurity     : str   = Field(..., example="No")
    OnlineBackup       : str   = Field(..., example="Yes")
    DeviceProtection   : str   = Field(..., example="No")
    TechSupport        : str   = Field(..., example="No")
    StreamingTV        : str   = Field(..., example="Yes")
    StreamingMovies    : str   = Field(..., example="Yes")
    Contract           : str   = Field(..., example="Month-to-month")
    PaperlessBilling   : str   = Field(..., example="Yes")
    PaymentMethod      : str   = Field(..., example="Electronic check")
    MonthlyCharges     : float = Field(..., example=70.35)
    TotalCharges       : float = Field(..., example=844.20)

# ─────────────────────────────────────────
# OUTPUT SCHEMA
# Defines exactly what the API returns
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction   : int
    churn_probability  : float
    risk_level         : str
    top_reasons        : list
    message            : str

# ─────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "🔴 High Risk"
    elif probability >= 0.4:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

# 1. Health Check
@app.get("/health")
def health_check():
    return {
        "status"  : "healthy ✅",
        "model"   : "XGBoost Churn Predictor",
        "version" : "1.0.0"
    }

# 2. Single Customer Prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Step 1: Convert input to DataFrame
        input_dict = customer.dict()
        input_df   = pd.DataFrame([input_dict])

        # Step 2: Preprocess
        input_transformed = preprocessor.transform(input_df)

        # Step 3: Predict
        churn_prob       = model.predict_proba(input_transformed)[0][1]
        churn_prediction = int(churn_prob >= 0.5)

        # Step 4: SHAP explanation
        shap_vals = explainer.shap_values(input_transformed)[0]

        # Get top 5 reasons
        feature_impacts = list(zip(all_feature_names, shap_vals))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        top_reasons = [
            {
                "feature" : feat,
                "impact"  : round(float(val), 4),
                "effect"  : "increases churn risk" if val > 0 else "decreases churn risk"
            }
            for feat, val in feature_impacts[:5]
        ]

        # Step 5: Build response
        risk_level = get_risk_level(churn_prob)
        message    = (
            "⚠️ This customer is likely to churn. Consider retention strategies."
            if churn_prediction == 1
            else "✅ This customer is likely to stay."
        )

        return PredictionResponse(
            churn_prediction  = churn_prediction,
            churn_probability = round(float(churn_prob), 4),
            risk_level        = risk_level,
            top_reasons       = top_reasons,
            message           = message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. Batch Prediction
@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    try:
        # Convert list of customers to DataFrame
        input_df        = pd.DataFrame([c.dict() for c in customers])
        input_transformed = preprocessor.transform(input_df)

        # Predict all at once
        churn_probs      = model.predict_proba(input_transformed)[:, 1]
        churn_predictions = (churn_probs >= 0.5).astype(int)

        results = []
        for i, (prob, pred) in enumerate(zip(churn_probs, churn_predictions)):
            results.append({
                "customer_index"   : i,
                "churn_prediction" : int(pred),
                "churn_probability": round(float(prob), 4),
                "risk_level"       : get_risk_level(prob)
            })

        return {
            "total_customers" : len(results),
            "high_risk_count" : sum(1 for r in results if r['churn_prediction'] == 1),
            "predictions"     : results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))