import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile

# -----------------------------
# Load trained model and scaler
# -----------------------------
model = joblib.load("models/DecisionTree.joblib")
scaler = joblib.load("models/scaler.joblib")
# -----------------------------
# Load dataset columns
# -----------------------------
zip_path = "data/archive.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:

    csv_file = [f for f in zip_ref.namelist() if f.endswith(".csv")][0]

    with zip_ref.open(csv_file) as file:
        df = pd.read_csv(file)

# Remove target columns
drop_columns = ["Class", "is_fraud"]

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Remove unwanted columns
feature_columns = [col for col in numeric_df.columns if col not in drop_columns]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details below:")

# Store user inputs
input_data = []

# Create input fields dynamically
for col in feature_columns:

    value = st.number_input(f"{col}", value=0.0)

    input_data.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Transaction"):

    input_array = np.array([input_data])
input_array = np.array([input_data])

# Scale data
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)

# Prediction probability
probability = model.predict_proba(input_scaled)[0][1]

if prediction[0] == 1:
    st.error("⚠ Fraudulent Transaction Detected")
    st.write(f"Fraud Probability: {probability:.2%}")

else:
    st.success("✅ Genuine Transaction")
    st.write(f"Fraud Probability: {probability:.2%}")