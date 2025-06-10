import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import joblib
import os
from scipy import stats

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


def run_causal_forest_with_params(
    X_normalized, T, Y, seeds, n_estimators, max_depth, min_samples_leaf
):
    mean_cates = []
    std_cates = []
    models = []

    for seed in seeds:
        model = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            ),
            model_t=RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            ),
            discrete_treatment=False,
            random_state=seed,
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
    mean_cate_std = np.mean(std_cates)

    return {
        "models": models,
        "mean_cates": mean_cates,
        "std_cates": std_cates,
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_cate_std": mean_cate_std,
    }


def run_causal_forest_optimization(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_optimized.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    seeds=range(10),
    param_grid=None,
):
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 15, 20],
            "min_samples_leaf": [5, 10, 20],
        }

    X, X_normalized, T, Y, scaler = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    results = []
    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for min_leaf in param_grid["min_samples_leaf"]:
                print(
                    f"Running with n_estimators={n_est}, max_depth={depth}, min_samples_leaf={min_leaf} ..."
                )
                res = run_causal_forest_with_params(
                    X_normalized, T, Y, seeds, n_est, depth, min_leaf
                )
                res.update(
                    {
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "min_samples_leaf": min_leaf,
                    }
                )
                results.append(res)
                print(
                    f"Mean CATE: {res['mean_of_means']:.4f}, "
                    f"Mean CATE Std: {res['mean_cate_std']:.4f}, "
                    f"T-test p-value: {res['p_value']:.4f}"
                )
                print("-" * 40)

    # Wähle Modell mit kleinstem mittlerem CATE-Standardabweichung (als Qualitätssignal)
    best = min(results, key=lambda x: x["mean_cate_std"])

    best_model = best["models"][-1]

    model_bundle = {
        "model": best_model,
        "features": X.columns.tolist(),
        "scaler": scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": best["mean_of_means"],
        "cate_std_mean": best["mean_cate_std"],
        "mean_cates_all_seeds": best["mean_cates"],
        "std_cates_all_seeds": best["std_cates"],
        "t_stat": best["t_stat"],
        "p_value": best["p_value"],
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "min_samples_leaf": best["min_samples_leaf"],
        "feature_importance": best_model.feature_importances_,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"\nBest model saved to: {model_path}")
    print(f"Best params: n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}")
    print(f"Mean estimated treatment effect (CATE): {best['mean_of_means']:.6f}")
    print(f"Mean std deviation of CATE estimates: {best['mean_cate_std']:.6f}")
    print(f"T-Test against zero: t = {best['t_stat']:.3f}, p = {best['p_value']:.3f}")

    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity:")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle


def get_trained_causal_forest_optimized():
    return run_causal_forest_optimization()


if __name__ == "__main__":
    bundle = get_trained_causal_forest_optimized()
    print("Model training and optimization complete.")
