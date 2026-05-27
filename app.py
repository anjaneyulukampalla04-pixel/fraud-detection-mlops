import streamlit as st
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
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Main Title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>
        💳 Credit Card Fraud Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About Project")

st.sidebar.write("""
This project predicts whether a transaction is fraudulent or genuine using Machine Learning.
""")

st.sidebar.success("Model: Decision Tree")
st.sidebar.image("plots/model_accuracy_barplot.png")

# -----------------------------
# Info Message
# -----------------------------
st.info("Enter transaction details below to predict fraud.")

# -----------------------------
# User Inputs
# -----------------------------
input_data = []

for col in feature_columns:

    value = st.number_input(
        f"{col}",
        value=0.0
    )

    input_data.append(value)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Transaction"):

    # Convert input to array
    input_array = np.array([input_data])

    # Scale data
    input_scaled = scaler.transform(input_array)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Fraud Prediction
    if prediction[0] == 1:

        st.error("⚠ Fraudulent Transaction Detected")

        st.metric(
            label="Fraud Probability",
            value=f"{probability:.2%}"
        )

    # Genuine Prediction
    else:

        st.success("✅ Genuine Transaction")

        st.balloons()

        st.metric(
            label="Fraud Probability",
            value=f"{probability:.2%}"
        )
        st.markdown("## Model Performance")

st.image("plots/roc_curves.png")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption("Developed using Streamlit and Machine Learning")