from fastapi import FastAPI
import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.preprocess import preprocess_input

# Initialize API
app = FastAPI()

# Load model and columns
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 

model = joblib.load(os.path.join(BASE_DIR, "models", "dt_model.pkl"))
trained_columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

# model = joblib.load("models/dt_model.pkl")
# trained_columns = joblib.load("models/columns.pkl")


@app.get("/")
def home():
    return {"message": "Insurance Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    API endpoint for prediction
    """

    # Preprocess input
    processed_data = preprocess_input(data, trained_columns)

    # Predict (log scale)
    pred_log = model.predict(processed_data)

    # Convert back to actual value
    prediction = float(np.exp(pred_log)[0])

    return {"predicted_charges": prediction}