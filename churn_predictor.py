import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("rf_churn_model_top10.pkl")

st.title("📉 Telco Customer Churn Predictor")
st.markdown("Input customer details to predict churn risk.")

# ✅ Full-width slider at the top
tenure = st.slider("Tenure (months)", 0, 72, 12)

# ✅ Two columns for the rest of the inputs
col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, step=10.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0, step=100.0)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    payment = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Convert inputs to one-hot encoded format
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0
}

input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df.reindex(columns=[
    'tenure',
    'TotalCharges',
    'MonthlyCharges',
    'InternetService_Fiber optic',
    'Contract_Two year',
    'PaymentMethod_Electronic check',
    'OnlineSecurity_Yes',
    'Contract_One year',
    'TechSupport_Yes',
    'PaperlessBilling_Yes'
])

# Predict
prediction = model.predict(input_df)
prediction_prob = model.predict_proba(input_df)

# Output
st.subheader("Prediction:")
if prediction[0] == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY.")

st.markdown(f"**Churn Probability:** {prediction_prob[0][1]:.2%}")
