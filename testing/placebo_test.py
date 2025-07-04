import numpy as np
import pandas as pd
from econml.grf import CausalForest
from data_extraction.utils import normalize_features, load_excel
from scipy import stats


def placebo_test(
    filepath="../data/fm_dem_sat_merged.xlsx",
    seeds=range(10)
):
    """
    Perform a placebo treatment test using an honest Causal Forest.

    Generate random placebo treatment values matching the original treatment's standard deviation,
    fit the honest Causal Forest on each seed, and record mean and std of estimated effects.

    Returns:
        mean_cates: List of mean treatment effects per seed
        std_cates: List of standard deviations of effects per seed
        t_stat: t-test statistic for mean effects vs zero
        p_value: p-value of the t-test
    """
    # Load panel data and impute missing numeric values
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))

    # Define features excluding treatment and outcome columns
    exclude = {"ZB_diff", "Mitglieder_diff_next", "Jahr", "Quartal", "Date"}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    # Normalize covariates
    X_norm, scaler = normalize_features(X)

    # Extract original treatment for scale reference
    original_treatment = df["ZB_diff"].values

    mean_cates = []
    std_cates = []

    # Loop over random seeds
    for seed in seeds:
        # Create placebo treatment with same standard deviation
        placebo_t = np.random.normal(loc=0, scale=np.std(original_treatment), size=len(df))

        # Initialize honest Causal Forest
        model = CausalForest(
            n_estimators=200,
            min_samples_leaf=5,
            max_depth=10,
            honest=True,
            max_features='sqrt',
            random_state=seed,
        )
        # Fit on placebo treatment
        model.fit(X_norm, placebo_t.reshape(-1, 1), df["Mitglieder_diff_next"].values)

        # Estimate treatment effects
        cates = model.predict(X_norm)
        mean_cates.append(np.mean(cates))
        std_cates.append(np.std(cates))

    # Aggregate results
    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    # Output summary
    print(f"Placebo Test Honest Causal Forest (over {len(seeds)} seeds):")
    print(f"Average Mean CATE           = {mean_of_means:.6f}")
    print(f"Std Dev of Mean CATEs       = {std_of_means:.6f}")
    print(f"T-Test: t = {t_stat:.3f}, p = {p_value:.3f}")
    print("Expectation: Mean CATE close to 0 and p-value > 0.05 indicates no placebo effect.")

    return mean_cates, std_cates, t_stat, p_value


if __name__ == "__main__":
    placebo_test()
