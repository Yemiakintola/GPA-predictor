import streamlit as st
import numpy as np
import joblib

# Load pipeline
pipeline = joblib.load("best_student_performance_pipeline.pkl")

st.title("Student Performance Predictor")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
parent_edu = st.selectbox("Parental Education", ["None", "Primary", "Secondary", "Tertiary"])
tutoring = st.selectbox("Tutoring", ["Yes", "No"])
support = st.selectbox("Parental Support", ["Yes", "No"])
extra = st.selectbox("Extracurricular", ["Yes", "No"])
sports = st.selectbox("Sports", ["Yes", "No"])
music = st.selectbox("Music", ["Yes", "No"])
volunteer = st.selectbox("Volunteering", ["Yes", "No"])
grade_class = st.selectbox("Grade Class", ["A", "B", "C", "D", "F"])

age = st.slider("Age", 10, 25)
study = st.slider("Study Time per Week (hours)", 0, 40)
absences = st.slider("Absences", 0, 100)

# Make prediction
input_data = pd.DataFrame([[
    gender, tutoring, support, extra, sports, music, volunteer,
    ethnicity, parent_edu, grade_class,
    age, study, absences
]], columns=[
    'Gender', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering',
    'Ethnicity', 'ParentalEducation', 'GradeClass',
    'Age', 'StudyTimeWeekly', 'Absences'
])

if st.button("Predict"):
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted GPA: {prediction[0]:.2f}")
