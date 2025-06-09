import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import os
import joblib

from data_extraction.utils import normalize_features

def prepare_data(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
):
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = pd.read_excel(filepath)

    df = df.fillna(df.median(numeric_only=True))
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])
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
    X, X_normalized, T, Y, scaler = prepare_data(filepath=filepath)

    results = []

    for max_depth in max_depths:
        for n_estimators in n_estimators_list:
            model = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                model_t=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                discrete_treatment=False,
                random_state=random_state
            )
            model.fit(Y, T, X=X_normalized)
            cates = model.effect(X_normalized)
            mean_cate = np.mean(cates)
            std_cate = np.std(cates)
            results.append({
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "mean_cate": mean_cate,
                "std_cate": std_cate
            })
            print(f"max_depth={max_depth}, n_estimators={n_estimators} => mean CATE={mean_cate:.4f}, std CATE={std_cate:.4f}")

    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    df_sensitivity = sensitivity_analysis()
    print("\nSummary of Sensitivity Analysis:")
    print(df_sensitivity)
