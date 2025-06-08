import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt

def run_causal_forest(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    categorical_col="Krankenkasse",
    plot_effects=True,
):
    # Load data
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = pd.read_excel(filepath)
    try:
        df = df.fillna(df.median())
    except Exception as e:
        print(e)

    # Ensure columns are string type for sklearn compatibility
    df.columns = df.columns.astype(str)

    # Treatment and outcome variables
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Select all columns except treatment, outcome, and categorical column as features
    feature_cols = df.columns.difference([treatment_col, outcome_col, categorical_col]).tolist()
    X = df[feature_cols]

    # One-hot encode the categorical column if it exists
    if categorical_col and categorical_col in df.columns:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        cat_encoded = encoder.fit_transform(df[[categorical_col]])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out([categorical_col]))
        # Reset index to align for concatenation
        X = pd.concat([X.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

    # Train-test split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )

    # Initialize Causal Forest model (without cache_values param)
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        discrete_treatment=False,  # Because ZB_diff is continuous
        random_state=42
    )

    # Fit the model
    model.fit(Y_train, T_train, X=X_train)

    # Estimate Conditional Average Treatment Effects (CATE)
    te_pred = model.effect(X_test)

    print("▶️ Mean estimated treatment effect (CATE):", np.mean(te_pred))

    # Feature importance for heterogeneity
    print("\n📊 Feature Importances (Heterogeneity):")
    for name, imp in zip(X.columns, model.feature_importances_):
        print(f"{name:30}: {imp:.4f}")

    # Tree interpretation shows which feature splits drive effect differences
    interpreter = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2)
    interpreter.interpret(model, X_test)
    interpreter.plot()

    # Histogram of estimated effects
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
    print("\n🟩 Sample CATEs:", te_pred[:5])  # Show first 5


if __name__ == "__main__":
    ca_fo()
