import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('voting_classifier.pkl')

# Title of the app
st.title("Campus Placement Prediction")

# Input fields for user to enter data
st.header("Enter Student Details")

# Numerical features
ssc_p = st.number_input("Secondary Education Percentage (10th)", min_value=0.0, max_value=100.0, value=70.0)
hsc_p = st.number_input("Higher Secondary Education Percentage (12th)", min_value=0.0, max_value=100.0, value=70.0)
degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
etest_p = st.number_input("Employability Test Percentage", min_value=0.0, max_value=100.0, value=70.0)
mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=70.0)

# Categorical features
gender = st.selectbox("Gender", ["Male", "Female"])
ssc_b = st.selectbox("Secondary Education Board", ["Central", "Others"])
hsc_b = st.selectbox("Higher Secondary Education Board", ["Central", "Others"])
hsc_s = st.selectbox("Higher Secondary Stream", ["Commerce", "Science", "Arts"])
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt"])
workex = st.selectbox("Work Experience", ["Yes", "No"])
specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])

# Convert categorical inputs to numerical
gender = 1 if gender == "Male" else 0
workex = 1 if workex == "Yes" else 0

# One-hot encoding for categorical features
input_data = {
    'ssc_p': ssc_p,
    'hsc_p': hsc_p,
    'degree_p': degree_p,
    'etest_p': etest_p,
    'mba_p': mba_p,
    'gender_Male': gender,
    'ssc_b_Others': 1 if ssc_b == "Others" else 0,
    'hsc_b_Others': 1 if hsc_b == "Others" else 0,
    'hsc_s_Science': 1 if hsc_s == "Science" else 0,
    'hsc_s_Commerce': 1 if hsc_s == "Commerce" else 0,
    'degree_t_Sci&Tech': 1 if degree_t == "Sci&Tech" else 0,
    'workex_Yes': workex,
    'specialisation_Mkt&Fin': 1 if specialisation == "Mkt&Fin" else 0
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Display the input data
st.subheader("Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Placement"):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("The student is likely to be **Placed**.")
    else:
        st.error("The student is likely to be **Not Placed**.")

    # Display prediction probabilities
    st.write(f"Probability of being Placed: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of being Not Placed: {prediction_proba[0][0]:.2f}")