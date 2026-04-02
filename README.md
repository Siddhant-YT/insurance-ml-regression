# Medical Insurance Cost Prediction

## Overview

This project is an end-to-end machine learning application that predicts
medical insurance charges based on user attributes such as age, BMI,
smoking status, and region.

The solution includes: - Data preprocessing and feature engineering -
Multiple regression models (Linear Regression, Decision Tree, Random
Forest, XGBoost, SVR) - Model evaluation and selection -
Production-ready inference pipeline - FastAPI-based backend for model
serving - Streamlit-based frontend for user interaction

The final selected model is a tuned Decision Tree Regressor based on
performance metrics.

------------------------------------------------------------------------

## Problem Statement

The objective is to predict individual medical insurance charges using
demographic and health-related features. This is a regression problem
where the target variable is continuous.

------------------------------------------------------------------------

## Dataset

The dataset used is the Medical Insurance dataset, which includes the
following features:

-   age: Age of the individual\
-   sex: Gender (male/female)\
-   bmi: Body Mass Index\
-   children: Number of dependents\
-   smoker: Smoking status (yes/no)\
-   region: Residential area (northeast, northwest, southeast,
    southwest)\
-   charges: Medical insurance cost (target variable)

------------------------------------------------------------------------

## Machine Learning Workflow

### 1. Data Preprocessing

-   Binary encoding for `sex` and `smoker`
-   One-hot encoding for `region`
-   Feature engineering: `bmi_age`
-   Log transformation of target (`charges`)

------------------------------------------------------------------------

### 2. Models Implemented

-   Linear Regression\
-   Decision Tree Regressor\
-   Random Forest Regressor\
-   XGBoost Regressor\
-   Support Vector Regressor (SVR)

------------------------------------------------------------------------

### 3. Evaluation Metrics

-   Mean Absolute Error (MAE)
-   Root Mean Squared Error (RMSE)
-   R-squared (R²)

------------------------------------------------------------------------

### 4. Model Selection

The tuned Decision Tree Regressor was selected as the best model based
on: - Lowest RMSE - Highest R² - Good generalization

------------------------------------------------------------------------

## Project Structure

insurance-ml-regression/
│
├── data/
│   └── insurance.csv
│
├── models/
│   ├── dt_model.pkl
│   └── columns.pkl
│
├── api/
│   └── app.py
│
├── ui/
│   └── streamlit_app.py
│
├── utils/
│   └── preprocess.py
│
├── requirements.txt
└── README.md
------------------------------------------------------------------------

## Inference Pipeline

-   Binary encoding for categorical variables\
-   Feature engineering\
-   One-hot encoding\
-   Column alignment\
-   Prediction\
-   Reverse log transformation

------------------------------------------------------------------------

## API (FastAPI)

### Endpoints

-   GET / → Health check\
-   POST /predict → Returns predicted charges

------------------------------------------------------------------------

## Streamlit UI

-   User input form\
-   API-based predictions\
-   Data visualizations

------------------------------------------------------------------------

## Installation

pip install -r requirements.txt

------------------------------------------------------------------------

## Run

Start API: uvicorn app:app --reload

Start UI: streamlit run streamlit_app.py

------------------------------------------------------------------------

## Key Learnings

-   Consistent preprocessing\
-   Model comparison\
-   Production pipeline\
-   API + UI integration

------------------------------------------------------------------------

## Future Improvements

-   Monitoring\
-   Docker\
-   Cloud deployment

------------------------------------------------------------------------

## Conclusion

This project demonstrates a complete ML lifecycle from training to
deployment.
