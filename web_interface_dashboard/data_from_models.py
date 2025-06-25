import joblib
import pandas as pd
from paper.test import starting_point
import statsmodels.api as sm
import numpy as np

def predict_data(insurer: str, zb_diff: float, modelpath: str) -> float:
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
    # Load base data for all insurers
    df = starting_point()

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

if __name__ == '__main__':
    # Example usage with causal forest model
    print(predict_data('aokbadenwürttemberg', 0.5, '../models/causal_forest_full_honest.pkl'))

    # Example usage with linear regression model
    print(predict_data('aokbadenwürttemberg', 0.5, '../models/lin_regression_model.pkl'))
