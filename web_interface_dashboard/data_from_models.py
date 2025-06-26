import joblib
import pandas as pd
import torch

from analysis_code.full_neural_network import NeuralNetwork, MODEL_PATH, SCALER_PATH, CATEGORICAL_COLS_PATH, SATISFACTION_COLS_PATH
from data_extraction.utils import column_name_cleanup, load_excel
from paper.test import starting_point
import statsmodels.api as sm
import numpy as np

def predict_data(df, insurer: str, zb_diff: float, modelpath: str) -> float:
    """
    Predict membership change for a given insurer and contribution change.
    Supports both causal forest and linear regression models saved in a unified format.

    Parameters:
    - insurer: Name of the insurer to predict for.
    - zb_diff: Contribution difference (ZB_diff) to apply.
    - modelpath: Path to the saved model bundle (joblib file).

    Returns:
    - Predicted membership change as a float.
    """
    # Load saved model bundle containing model, scaler, features and flags
    model_bundle = joblib.load(modelpath)
    model = model_bundle['model']
    scaler = model_bundle.get('scaler', None)           # Optional scaler
    features = model_bundle['features']                  # Feature names used in training
    add_const = model_bundle.get('add_constant', False)  # Flag for adding intercept (for OLS)

    # Filter data for the specified insurer and set the contribution difference
    row = df[df['Krankenkasse'] == insurer].copy()
    if row.empty:
        raise ValueError(f"Insurer '{insurer}' not found in input data.")
    if modelpath == '../models/did_stratified_model.pkl':
        row['treatment']= zb_diff
    #elif modelpath == '../models/metaregression_model.pkl':
        #add_const = False
        #row['const'] = zb_diff
    else:
        row['ZB_diff'] = zb_diff
    # Select only relevant features from the row
    X = row[features]

    # Apply scaling if scaler is present (e.g. for ML models)
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=features, index=X.index)

    # Add constant/intercept column for linear regression models if needed
    if add_const:
        X = sm.add_constant(X, has_constant='add')

    # Perform prediction
    pred = model.predict(X)

    # Extract single scalar value safely from prediction output (array or Series)
    if isinstance(pred, (pd.Series, np.ndarray)):
        pred = pred.item()
    else:
        pred = float(pred)

    return pred
def full_pred(insurers=['aokbadenwürttemberg'], zb_diff=0.1):
    """
    Runs all four methods (CF, LR, DiD, Meta-regression) for each insurer
    and returns a DataFrame with insurers as rows and methods as columns.
    """
    # Prepare empty result
    methods = ['cf', 'lr', 'did', 'meta']
    df_result = pd.DataFrame(index=insurers, columns=methods, dtype=float)

    #Causal Forest
    for insurer in insurers:
        df = starting_point()
        df_result.loc[insurer, 'cf'] = predict_data(
            df, insurer, zb_diff, '../models/causal_forest_full_honest.pkl'
        )

    #Linear Regression
    for insurer in insurers:
        df = starting_point()
        df_result.loc[insurer, 'lr'] = predict_data(
            df, insurer, zb_diff, '../models/lin_regression_model.pkl'
        )

    #Stratified DiD
    for insurer in insurers:
        treatment_var = 'ZB_diff'
        time_var = 'Date'
        df = starting_point(is_did=True)
        # df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
        df['treatment'] = (df[treatment_var] != 0).astype(int)

        # Determine first treatment time per insurer
        first_treat = df[df['treatment'] == 1].groupby('Krankenkasse')[time_var].min()
        df = df.join(first_treat.rename('first_treat'), on='Krankenkasse')
        df['post'] = (df[time_var] >= df['first_treat']).astype(int)

        # Fill missing numeric values with median
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        df = column_name_cleanup(df)
        df_copy = df.copy()
        df_copy['treatment:post'] = df_copy['treatment'] * df['post']

        df_result.loc[insurer, 'did'] = predict_data(
            df_copy, insurer, zb_diff, '../models/did_stratified_model.pkl'
        )

    #Meta-Regression (use pre-stored slopes)
    bundle = joblib.load("../models/metaregression_model.pkl")
    slopes = bundle["slopes"]  # dict: { insurer: theta_i, ... }
    for insurer in insurers:
        if insurer not in slopes:
            raise ValueError(f"Insurer '{insurer}' not found in slopes")
        df_result.loc[insurer, 'meta'] = slopes[insurer] * zb_diff

    return df_result


def pred_nn(insurer: str, zb_diff: float):
    """
    Predict churn via the trained Neural Network.
    """
    # Get base row
    df = starting_point()
    row = df[df['Krankenkasse'] == insurer].copy()
    if row.empty:
        raise ValueError(f"Insurer '{insurer}' not found")
    row['ZB_diff'] = zb_diff

    # Load artifacts
    scaler = joblib.load(SCALER_PATH)
    cat_cols = joblib.load(CATEGORICAL_COLS_PATH)
    sat_cols = joblib.load(SATISFACTION_COLS_PATH)

    #One-hot encode
    row_enc = pd.get_dummies(row, columns=cat_cols, drop_first=True)
    for c in sat_cols:
        if c not in row_enc.columns:
            row_enc[c] = 0
    X_df = row_enc[sat_cols].astype(float)

    # 4) Scale & tensorize
    X_scaled = scaler.transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 5) Load model & predict
    model = NeuralNetwork(input_size=X_scaled.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).item()

    # 6) Convert percent back to fraction
    print(y_pred / 100.0)



if __name__ == '__main__':
    print(full_pred())
