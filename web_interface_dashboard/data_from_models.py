import joblib
import pandas as pd
import torch
import sys
import os


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis_code.full_neural_network import NeuralNetwork, MODEL_PATH, SCALER_PATH, CATEGORICAL_COLS_PATH, SATISFACTION_COLS_PATH
from data_extraction.utils import column_name_cleanup
from paper.test import starting_point
import statsmodels.api as sm
import numpy as np
from analysis_code.full_neural_network import train_and_save_neural_network

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


def pred_nn(insurer: str='aokbadenwürttemberg', zb_diff: float=0.1):
    """
    Predict churn via the trained Neural Network.
    """
    print("predicting churn via the trained Neural Network...")
    # Load necessary artifacts, train and save if missing
    if not (os.path.exists(SCALER_PATH) and os.path.exists(CATEGORICAL_COLS_PATH) and os.path.exists(SATISFACTION_COLS_PATH) and os.path.exists(MODEL_PATH)):
        print("Model has to be trained.")
        train_and_save_neural_network()
    scaler = joblib.load(SCALER_PATH)
    categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)
    satisfaction_columns = joblib.load(SATISFACTION_COLS_PATH)
    print("loading worked")
    # Load the dataset and filter for the specified insurer
    df = starting_point()
    row = df[df['Krankenkasse'] == insurer].copy()
    if row.empty:
        raise ValueError(f"Insurer '{insurer}' not found in input data.")

    # Set zb_diff and average values for other columns
    row['ZB_diff'] = zb_diff
    """
    print(row)
    for col in satisfaction_columns:
        if col not in ['ZB_diff'] and col in df.columns:
            row[col] = df[col].mean()
    """

    # One-hot encode and ensure all columns exist
    row = pd.get_dummies(row, columns=categorical_cols, drop_first=True)
    missing_cols = [col for col in satisfaction_columns if col not in row.columns]
    if missing_cols:
        row = pd.concat([row, pd.DataFrame(0, index=row.index, columns=missing_cols)], axis=1)

    # Prepare input data
    X_input = row[satisfaction_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    X_input_scaled = scaler.transform(X_input)
    X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)

    # Load the trained neural network model
    input_size = X_input.shape[1]
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Perform prediction
    with torch.no_grad():
        predictions = model(X_input_tensor).numpy().flatten()
    #print(row['Mitglieder'].to_numpy()[0])
    #print(predictions[0]/100)
    print("done")
    return ((predictions[0]/100) *row['Mitglieder'].to_numpy()[0])/100




def full_pred(insurers=['aokbadenwürttemberg','aokplus'], zb_diff=0.1):
    """
    Runs all four methods (CF, LR, DiD, Meta-regression) for each insurer
    and returns a DataFrame with insurers as rows and methods as columns.
    """
    # Prepare empty result
    methods = ['cf', 'lr', 'did', 'meta', 'nn']
    df_result = pd.DataFrame(index=insurers, columns=methods, dtype=float)
    print("Checking if models exist.")
    try:
        joblib.load('../models/causal_forest_full_honest.pkl')
        print("Causal Forest loaded successfully.")
    except FileNotFoundError:
        print("Causal Forest model filenot found. Training new model...")
        from analysis_code.cf_honest_trees import run_causal_forest_crossfit
        run_causal_forest_crossfit()
    try:
        joblib.load('../models/lin_regression_model.pkl')
        print("Linear Regression loaded successfully.")
    except FileNotFoundError:
        print("Linear Regression model file not found. Training new model...")
        from analysis_code.linear_regression import regression_fm_adj_r2
        regression_fm_adj_r2(cv=5)
    try:
        joblib.load('../models/did_stratified_model.pkl')
        print("DiD loaded successfully.")
    except FileNotFoundError:
        print("DiD model file not found. Training new model...")
        from analysis_code.difference_in_difference import did
        did()
    try:
        joblib.load('../models/metaregression_model.pkl')
        print("Metaregression loaded successfully.")
    except FileNotFoundError:
        print("Metregression model file not found. Training new model...")
        from analysis_code.heterogeneous_treatment_pipeline import slope_meta
        slope_meta()
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

    for insurer in insurers:
        try:
            df_result.loc[insurer, 'nn'] = pred_nn(insurer,zb_diff)
        except KeyError:
            print("KeyError")
            print(insurer)
            df_result.loc[insurer, 'nn'] =0

    return df_result





if __name__ == '__main__':
    #print(full_pred(['aokbadenwürttemberg','aokplus']))
    print(pred_nn(insurer='aokplus'))
    print(pred_nn(insurer='aokbadenwürttemberg'))
    print(pred_nn(insurer='bkkpwc'))
