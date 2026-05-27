import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model and scaler
# -----------------------------
model = joblib.load("outputs/DecisionTree.joblib")
scaler = joblib.load("outputs/scaler.joblib")

# -----------------------------
# Feature columns
# -----------------------------
feature_columns = [
    "amt",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>
    💳 Credit Card Fraud Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.info("Enter transaction details below to predict fraud.")
st.sidebar.title("About Project")

st.sidebar.write("""
This project predicts whether a transaction is fraudulent or genuine using Machine Learning.
""")

st.sidebar.success("Model: Decision Tree")

# Store user inputs
input_data = []

# Create input fields dynamically
for col in feature_columns:

    value = st.number_input(f"{col}", value=0.0)

    input_data.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Transaction"):

    input_array = np.array([input_data])

    # Scale data
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
        st.write(f"Fraud Probability: {probability:.2%}")

    else:
        st.success("✅ Genuine Transaction")
        st.write(f"Fraud Probability: {probability:.2%}")

        st.markdown("---")
st.caption("Developed using Streamlit and Machine Learning")