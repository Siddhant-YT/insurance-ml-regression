import pandas as pd

def preprocess_input(input_data, trained_columns):
    """
    This function replicates training preprocessing steps
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Binary encoding (same as training)
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # Feature engineering
    df['bmi_age'] = df['bmi'] * df['age']

    # One-hot encoding for region
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    # Align columns with training data
    df = df.reindex(columns=trained_columns, fill_value=0)

    return df