import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt


def run_causal_forest(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    features=None,
    categorical_col="Krankenkasse",
    plot_effects=True,
):
    """
    Runs a Causal Forest estimation using econml's CausalForestDML.

    Parameters:
    - filepath: path to the Excel file containing the data
    - treatment_col: column name for the treatment (e.g., ZB_diff)
    - outcome_col: column name for the outcome (e.g., Mitglieder_diff_next)
    - features: list of column names used as control variables (X)
    - categorical_col: name of a categorical column to one-hot encode (e.g., Krankenkasse)
    - plot_effects: whether to plot the distribution of estimated treatment effects

    Returns:
    - model: fitted CausalForestDML model
    - te_pred: array of estimated treatment effects on the test set
    """

    # Load the data
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = pd.read_excel(filepath)

    # Define treatment and outcome
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Define feature columns
    if features is None:
        features = ["Mitglieder", "Zusatzbeitrag", "Versicherte"]  # Default fallback

    X = df[features]

    # One-hot encode categorical variable (if exists)
    if categorical_col and categorical_col in df.columns:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        cat_encoded = encoder.fit_transform(df[[categorical_col]])
        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(cat_encoded)], axis=1)

    # Train-test split for cross-fitting
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )

    # Initialize the Causal Forest model
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10),
        discrete_treatment=False,
        random_state=42,
    )

    # Fit the model
    model.fit(Y_train, T_train, X=X_train)

    # Predict treatment effects on the test set
    te_pred = model.effect(X_test)

    print("Mean estimated treatment effect (CATE):", np.mean(te_pred))

    # Optional: plot histogram of treatment effects
    if plot_effects:
        plt.figure(figsize=(8, 5))
        plt.hist(te_pred, bins=30, edgecolor="black")
        plt.title("Distribution of Estimated Treatment Effects (CATE)")
        plt.xlabel("Treatment Effect")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    return model, te_pred

def ca_fo():
    model, te_pred = run_causal_forest()
    print(te_pred)
    print(model.summary())
ca_fo()