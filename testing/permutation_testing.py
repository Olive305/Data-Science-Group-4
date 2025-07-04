import numpy as np
import pandas as pd
from econml.grf import CausalForest
from data_extraction.utils import normalize_features, load_excel
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
    """
    Perform a permutation test to evaluate the null distribution of average treatment effects
    estimated by an honest Causal Forest.

    The function fits the forest on real data, records the mean Conditional Average Treatment Effect (CATE),
    then repeatedly permutes the treatment assignments to build a null distribution of mean CATEs.

    Returns:
        real_mean_cate: Observed average CATE on original data
        null_distribution: Array of mean CATEs under permuted treatments
        p_value: Proportion of permuted mean CATEs as extreme as the real one
    """
    # Set random seed
    np.random.seed(seed)

    # Load data (Excel) and impute numeric missing values
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))

    # Select numeric covariates excluding treatment and outcome
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    # Normalize covariates
    X_norm, _ = normalize_features(X)

    # Extract outcome and treatment vectors
    Y = df[outcome_col].values
    T_original = df[treatment_col].values

    # Fit honest Causal Forest on real data
    real_model = CausalForest(
        n_estimators=200,
        min_samples_leaf=5,
        max_depth=10,
        honest=True,
        max_features='sqrt',
        random_state=seed,
    )
    real_model.fit(X_norm, T_original.reshape(-1, 1), Y)
    real_cate = real_model.predict(X_norm)
    real_mean_cate = np.mean(real_cate)

    # Build null distribution via treatment permutation
    null_distribution = []
    for i in range(n_permutations):
        T_perm = np.random.permutation(T_original)
        perm_model = CausalForest(
            n_estimators=200,
            min_samples_leaf=5,
            max_depth=10,
            honest=True,
            max_features='sqrt',
            random_state=seed + i,
        )
        perm_model.fit(X_norm, T_perm.reshape(-1, 1), Y)
        perm_cate = perm_model.predict(X_norm)
        null_distribution.append(np.mean(perm_cate))

    null_distribution = np.array(null_distribution)

    # Compute two-sided p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(real_mean_cate))

    # Output summary
    print("=== Permutation Test Honest Causal Forest ===")
    print(f"Real Mean CATE            = {real_mean_cate:.6f}")
    print(f"Null Mean CATE (mean)     = {np.mean(null_distribution):.6f}")
    print(f"P-Value                   = {p_value:.4f} (>=0.05 suggests no effect under null)")
    print("==============================================")

    return real_mean_cate, null_distribution, p_value


if __name__ == "__main__":
    permutation_test_causal_forest()
