import pandas as pd
import numpy as np
from econml.grf import CausalForest
import os
import joblib

# Ensure required packages are installed
os.system("pip install pandas numpy scikit-learn econml openpyxl joblib")

from data_extraction.utils import normalize_features


def prepare_data(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
):
    """
    Load panel data, impute missing values, normalize features, and extract treatment/outcome arrays.

    Returns:
        X: Original numeric feature matrix
        X_normalized: Normalized feature matrix
        T: Treatment vector
        Y: Outcome vector
        scaler: Fitted normalization object
    """
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = pd.read_excel(filepath)

    # Impute missing numeric values with column medians
    df = df.fillna(df.median(numeric_only=True))

    # Exclude non-feature columns
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    # Normalize features
    X_normalized, scaler = normalize_features(X)
    T = df[treatment_col].values
    Y = df[outcome_col].values

    return X, X_normalized, T, Y, scaler


def sensitivity_analysis(
    filepath="../data/fm_dem_sat_merged.xlsx",
    max_depths=[5, 10, 15],
    n_estimators_list=[50, 100, 200],
    random_state=42
):
    """
    Conduct sensitivity analysis by varying Causal Forest hyperparameters.

    For each combination of max_depth and number of trees, fit an honest Causal Forest,
    compute mean and standard deviation of Conditional Average Treatment Effects (CATEs),
    and collect results.

    Returns:
        DataFrame summarizing mean and std of CATEs for each hyperparameter setting.
    """
    X, X_norm, T, Y, scaler = prepare_data(filepath=filepath)

    results = []

    # Iterate over hyperparameter grid
    for max_depth in max_depths:
        for n_estimators in n_estimators_list:
            # Initialize honest Causal Forest with specified parameters
            model = CausalForest(
                n_estimators=n_estimators,
                min_samples_leaf=5,
                max_depth=max_depth,
                honest=True,
                max_features='sqrt',
                random_state=random_state
            )
            # Fit model on treatment and outcome data
            model.fit(X_norm, T.reshape(-1, 1), Y)
            # Estimate individualized treatment effects
            cates = model.predict(X_norm)
            mean_cate = np.mean(cates)
            std_cate = np.std(cates)

            # Store results for this hyperparameter setting
            results.append({
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "mean_cate": mean_cate,
                "std_cate": std_cate
            })
            print(f"max_depth={max_depth}, n_estimators={n_estimators} => mean CATE={mean_cate:.4f}, std CATE={std_cate:.4f}")

    # Compile results into DataFrame
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    df_sensitivity = sensitivity_analysis()
    print("\nSummary of Sensitivity Analysis:")
    print(df_sensitivity)
