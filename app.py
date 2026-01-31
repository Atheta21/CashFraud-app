import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details below:")

# Create two columns for cleaner UI
col1, col2 = st.columns(2)

inputs = []

with col1:
    time = st.number_input("Time", value=0.0)
    inputs.append(time)

    for i in range(1, 15):
        val = st.number_input(f"V{i}", value=0.0)
        inputs.append(val)

with col2:
    for i in range(15, 29):
        val = st.number_input(f"V{i}", value=0.0)
        inputs.append(val)

    amount = st.number_input("Amount", value=0.0)
    inputs.append(amount)

# Predict Button
if st.button("🔍 Check Transaction"):

    data = np.array([inputs])   # Shape (1,30)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Result:")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: **{probability:.4f}**")