import numpy as np
import pandas as pd
from econml.grf import CausalForest
from data_extraction.utils import normalize_features
from scipy import stats


def lead_effect_test(filepath="../data/fm_dem_sat_merged.xlsx", seeds=range(10)):
    """
    Perform a placebo lead-effect test using an honest Causal Forest on the previous-period outcome.

    For each random seed, fit an honest Causal Forest to estimate the causal effect of the treatment
    (ZB_diff) on the lead outcome (Mitglieder_diff_prev). Then calculate mean and std of CATEs across seeds.

    Returns:
        mean_cates: List of mean treatment effects per seed
        std_cates: List of std deviations of treatment effects per seed
        t_stat: t-test statistic comparing mean treatment effects to zero
        p_value: p-value of the t-test
    """
    # Load panel data and impute missing values
    df = pd.read_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))

    # Sort and compute lead outcome: change from next period back to current
    df = df.sort_values(by=["Krankenkasse", "Date"])
    df['Mitglieder_diff_prev'] = df.groupby('Krankenkasse')['Mitglieder'].shift(+1) - df['Mitglieder']
    df = df.dropna(subset=['Mitglieder_diff_prev'])

    outcome_col_prev = 'Mitglieder_diff_prev'
    treatment_col = 'ZB_diff'

    # Select numeric covariates excluding treatment and outcomes
    exclude = {treatment_col, outcome_col_prev, 'Jahr', 'Quartal', 'Date', 'Mitglieder_diff_next'}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    # Normalize features
    X_norm, scaler = normalize_features(X)
    Y_lead = df[outcome_col_prev].values
    T = df[treatment_col].values

    mean_cates = []
    std_cates = []

    # For each random seed, fit an honest Causal Forest and record summary
    for seed in seeds:
        model = CausalForest(
            n_estimators=200,
            min_samples_leaf=5,
            max_depth=10,
            honest=True,
            max_features='sqrt',
            random_state=seed,
        )
        # Fit model to lead outcome
        model.fit(X_norm, T.reshape(-1, 1), Y_lead)
        # Estimate individualized treatment effects
        cates = model.predict(X_norm)

        # Compute mean and std of CATEs for this seed
        mean_cates.append(np.mean(cates))
        std_cates.append(np.std(cates))

    # Aggregate across seeds
    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    # Print lead-effect test results
    print(f"Lead Effect Test (over {len(seeds)} seeds):")
    print(f"Average Mean CATE      = {mean_of_means:.6f}")
    print(f"Std Dev of Mean CATEs  = {std_of_means:.6f}")
    print(f"T-Test: t = {t_stat:.3f}, p = {p_value:.3f}")
    print("Expectation: Mean CATE near 0 and p-value > 0.05 indicates no lead effects.")

    return mean_cates, std_cates, t_stat, p_value


if __name__ == "__main__":
    seeds = range(10)
    lead_effect_test(seeds=seeds)
