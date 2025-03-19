# importing the necessary libraries
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("logistic_regression_model.pkl")

st.title("Fraud Detection App")

# Create input fields for all 30 features
features = []
for i in range(30):
    value = st.number_input(f"Enter Feature {i+1}", value=0.0)
    features.append(value)

# Convert input into numpy array
input_data = np.array([features]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
