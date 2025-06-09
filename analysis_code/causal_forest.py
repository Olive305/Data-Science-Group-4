import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import joblib
import os

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

def run_causal_forest_final(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_full.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
):
    X, X_normalized, T, Y, scaler = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        discrete_treatment=False,
        random_state=42
    )
    model.fit(Y, T, X=X_normalized)

    cates = model.effect(X_normalized)
    cate_std = model.effect_interval(X_normalized)[1] - cates
    mean_cate = np.round(np.mean(cates), 4)
    mean_std = np.round(np.mean(cate_std), 4)

    model_bundle = {
        "model": model,
        "features": X.columns.tolist(),
        "scaler": scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": mean_cate,
        "cate_std_mean": mean_std,
        "feature_importance": model.feature_importances_,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)
    print(f"Full model bundle saved to: {model_path}")
    print(f"Mean estimated treatment effect (CATE) per 1% change: {mean_cate}")
    print(f"Mean standard deviation of CATE estimates: {mean_std}")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity:")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle

def get_trained_causal_forest():
    return run_causal_forest_final()

if __name__ == "__main__":
    bundle = get_trained_causal_forest()
    print("Model training complete.")
