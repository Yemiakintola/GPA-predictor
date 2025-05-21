import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('student_performance_predictor.pkl')

# Title of the app
st.title("Student Performance Predictor")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
parent_education = st.selectbox("Parental Education Level", [
    "None", "Primary", "Secondary", "Tertiary"
])
study_time = st.slider("Study Time (hours per day)", 0, 10)
absences = st.slider("Number of Absences", 0, 100)
internet = st.selectbox("Internet Access at Home", ["Yes", "No"])

# Encoding categorical variables (same as training)
gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet == "Yes" else 0

education_map = {
    "None": 0,
    "Primary": 1,
    "Secondary": 2,
    "Tertiary": 3
}
parent_education_encoded = education_map[parent_education]

# Prepare input data
input_data = np.array([[gender_encoded, parent_education_encoded, study_time, absences, internet_encoded]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Performance: {prediction[0]:.2f}")
