import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt

from data_extraction.utils import normalize_features


def run_causal_forest(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    plot_effects=True,
):
    # Load data
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = pd.read_excel(filepath)

    # Fill missing numeric values with column medians
    df = df.fillna(df.median(numeric_only=True))

    # Ensure all column names are strings
    df.columns = df.columns.astype(str)

    # Define treatment and outcome variables
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Use all numeric features except treatment and outcome
    feature_cols = df.columns.difference([treatment_col, outcome_col]).tolist()
    X = df[feature_cols]
    X = X.select_dtypes(include=[np.number])

    X, scaler = normalize_features(X)

    # Train-test split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )

    # Initialize causal forest model
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        discrete_treatment=False,
        random_state=42
    )

    # Fit model to data
    model.fit(Y_train, T_train, X=X_train)

    # Estimate Conditional Average Treatment Effects (CATE)
    te_pred = model.effect(X_test)

    # Output mean CATE
    print("Mean estimated treatment effect (CATE):", np.mean(te_pred))

    # Display feature importances for treatment effect heterogeneity
    print("\nFeature importances (treatment effect heterogeneity):")
    for name, imp in zip(X_train.columns, model.feature_importances_):
        print(f"{name:30}: {imp:.4f}")

    # Visualize decision rules for heterogeneity (depth-2 tree)
    interpreter = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2)
    interpreter.interpret(model, X_test)
    interpreter.plot()

    # Histogram of estimated treatment effects
    if plot_effects:
        plt.figure(figsize=(8, 5))
        plt.hist(te_pred, bins=30, edgecolor="black")
        plt.title("Distribution of Estimated Treatment Effects (CATE)")
        plt.xlabel("Treatment Effect")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    return model, te_pred, scaler


def ca_fo():

    model, te_pred, scaler = run_causal_forest()
    print("\nSample CATEs:", te_pred[:5])


if __name__ == "__main__":
    ca_fo()
