import streamlit as st
import requests, os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Insurance Cost Predictor")

st.title("Medical Insurance Cost Prediction")

st.write("""
This application predicts medical insurance charges based on user details.
The model is trained on features like age, BMI, smoking status, etc.
""")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Details")

age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_data = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Charges"):

    # Call API
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    if response.status_code == 200:
        prediction = response.json()["predicted_charges"]
        st.success(f"Predicted Insurance Charges: {round(prediction, 2)}")
    else:
        st.error("Error in prediction API")


# # -----------------------------
# # Data Insights Section
# # -----------------------------
# st.subheader("Dataset Insights")

# # Load dataset (for visualization)
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # points to 3_Regression_ML/
# df = pd.read_csv(os.path.join(BASE_DIR, "data", "insurance.csv"))

# # df = pd.read_csv("data/insurance.csv")

# # Plot 1: Charges distribution
# fig1, ax1 = plt.subplots()
# ax1.hist(df["charges"], bins=30)
# ax1.set_title("Distribution of Insurance Charges")
# st.pyplot(fig1)

# # Plot 2: Charges vs Age
# fig2, ax2 = plt.subplots()
# ax2.scatter(df["age"], df["charges"])
# ax2.set_title("Age vs Charges")
# st.pyplot(fig2)

# # Plot 3: Smoker vs Charges
# fig3, ax3 = plt.subplots()
# df.boxplot(column="charges", by="smoker", ax=ax3)
# ax3.set_title("Charges by Smoker Status")
# st.pyplot(fig3)