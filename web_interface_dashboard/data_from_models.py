import joblib
import pandas as pd

from data_extraction.utils import column_name_cleanup
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

def pred_cf():
    df = starting_point()
    print(predict_data(df,'aokbadenwürttemberg', 0.5, '../models/causal_forest_full_honest.pkl'))

def pred_lr():
    df = starting_point()
    print(predict_data(df,'aokbadenwürttemberg', 0.5, '../models/lin_regression_model.pkl'))

def pred_did(treatment_var='ZB_diff', time_var='Date'):
    df = starting_point(is_did=True)
    #df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
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
    df_copy['treatment:post'] = df_copy['treatment']*df['post']
    print(predict_data(df_copy,'aokbadenwürttemberg', 0.5, '../models/did_stratified_model.pkl'))

def pred_nn():
    df = starting_point()


if __name__ == '__main__':
    pred_did()
