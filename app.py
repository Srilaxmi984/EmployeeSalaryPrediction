import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AI Salary Predictor", layout="wide")

# Load model
model = pickle.load(open("salary_model.pkl","rb"))
le_gender, le_education, le_job = pickle.load(open("encoders.pkl","rb"))

# Title
st.title("💼 AI Employee Salary Prediction Dashboard")
st.write("Predict employee salary using Machine Learning")

# Sidebar inputs
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age",18,65)

gender = st.sidebar.selectbox(
    "Gender",
    le_gender.classes_
)

education = st.sidebar.selectbox(
    "Education Level",
    le_education.classes_
)

job = st.sidebar.selectbox(
    "Job Title",
    le_job.classes_
)

experience = st.sidebar.slider(
    "Years of Experience",
    0,40
)

# Encode inputs
gender_enc = le_gender.transform([gender])[0]
edu_enc = le_education.transform([education])[0]
job_enc = le_job.transform([job])[0]

# Layout columns
col1, col2 = st.columns(2)

# Prediction column
with col1:

    st.subheader("Salary Prediction")

    if st.button("Predict Salary 💰"):

        features = np.array([[age, gender_enc, edu_enc, job_enc, experience]])

        prediction = model.predict(features)

        st.success(f"Estimated Salary: ₹ {int(prediction[0])}")

# Visualization column
with col2:

    st.subheader("Dataset Visualization")

    data = pd.read_csv("dataset/Salary_Data.csv")

    fig, ax = plt.subplots()

    ax.scatter(data["Years of Experience"], data["Salary"])

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    st.pyplot(fig)

# Dataset preview
st.subheader("Dataset Preview")

data = pd.read_csv("dataset/Salary_Data.csv")

st.dataframe(data.head(10))