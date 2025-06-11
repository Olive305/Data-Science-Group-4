import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import joblib
import os
from scipy import stats

from data_extraction.utils import normalize_features, load_excel


def prepare_data(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
):
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)

    df = df.fillna(df.median(numeric_only=True))
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])
    X_normalized, scaler = normalize_features(X)
    T = df[treatment_col].values
    Y = df[outcome_col].values

    return X, X_normalized, T, Y, scaler


def run_causal_forest_repeated(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_full.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    seeds=range(10),
):
    X, X_normalized, T, Y, scaler = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    mean_cates = []
    std_cates = []
    models = []

    for seed in seeds:
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            discrete_treatment=False,
            random_state=seed
        )
        model.fit(Y, T, X=X_normalized)

        cates = model.effect(X_normalized)
        cate_std = model.effect_interval(X_normalized)[1] - cates

        mean_cates.append(np.mean(cates))
        std_cates.append(np.mean(cate_std))
        models.append(model)

    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    # Policy Value berechnen mit dem letzten Modell
    final_cates = models[-1].effect(X_normalized)
    policy = final_cates > 0  # Policy: Behandle nur, wenn positiver Effekt
    policy_value = np.mean(Y[policy])

    model_bundle = {
        "model": models[-1],
        "features": X.columns.tolist(),
        "scaler": scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": mean_of_means,
        "cate_std_mean": np.mean(std_cates),
        "feature_importance": models[-1].feature_importances_,
        "mean_cates_all_seeds": mean_cates,
        "std_cates_all_seeds": std_cates,
        "t_stat": t_stat,
        "p_value": p_value,
        "policy_value": policy_value,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"Full model bundle saved to: {model_path}")
    print(f"Mean estimated treatment effect (CATE) over seeds: {mean_of_means:.6f}")
    print(f"Std deviation of mean CATEs over seeds: {std_of_means:.6f}")
    print(f"T-Test against zero: t = {t_stat:.3f}, p = {p_value:.3f}")
    print(f"Mean standard deviation of CATE estimates: {np.mean(std_cates):.6f}")
    print(f"Policy Value (expected outcome from model policy): {policy_value:.6f}")

    importances = pd.Series(models[-1].feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity:")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle



def get_trained_causal_forest():
    return run_causal_forest_repeated()


if __name__ == "__main__":
    bundle = get_trained_causal_forest()
    print("Model training complete.")
