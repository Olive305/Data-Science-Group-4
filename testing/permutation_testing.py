import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from data_extraction.utils import normalize_features
from scipy import stats

def permutation_test_causal_forest(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    n_permutations=100,
    seed=42,
):
    np.random.seed(seed)
    df = pd.read_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))

    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])
    X_norm, _ = normalize_features(X)

    Y = df[outcome_col].values
    T_original = df[treatment_col].values

    real_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
        discrete_treatment=False,
        random_state=seed,
    )
    real_model.fit(Y, T_original, X=X_norm)
    real_cate = real_model.effect(X_norm)
    real_mean_cate = np.mean(real_cate)

    null_distribution = []

    for i in range(n_permutations):
        T_perm = np.random.permutation(T_original)
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed + i),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed + i),
            discrete_treatment=False,
            random_state=seed + i,
        )
        model.fit(Y, T_perm, X=X_norm)
        cates = model.effect(X_norm)
        mean_cate = np.mean(cates)
        null_distribution.append(mean_cate)

    null_distribution = np.array(null_distribution)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(real_mean_cate))

    print("=== Permutation Test ===")
    print(f"Real Mean CATE: {real_mean_cate:.4f}")
    print(f"Null Mean CATE (mean of permuted): {np.mean(null_distribution):.4f}")
    print(f"P-Value: {p_value:.4f} (should be > 0.05 under null hypothesis)")
    print("========================")

    return real_mean_cate, null_distribution, p_value

if __name__ == "__main__":
    permutation_test_causal_forest()