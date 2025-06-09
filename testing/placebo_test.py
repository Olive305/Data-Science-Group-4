import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from data_extraction.utils import normalize_features
from scipy import stats

def placebo_test(filepath="../data/fm_dem_sat_merged.xlsx", seeds=range(10)):
    # Load data
    df = pd.read_excel(filepath)
    # Fill missing values with median (numeric columns only)
    df = df.fillna(df.median(numeric_only=True))

    # Define columns to exclude from features
    exclude = {"ZB_diff", "Mitglieder_diff_next", "Jahr", "Quartal", "Date"}
    # Select features (numeric only)
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    # Normalize features using your project's normalization function
    X_norm, scaler = normalize_features(X)
    # Outcome variable
    Y = df["Mitglieder_diff_next"].values
    # Original treatment variable (for scale reference)
    original_treatment = df["ZB_diff"].values

    mean_cates = []
    std_cates = []

    # Run placebo test over multiple random seeds to assess stability
    for seed in seeds:
        # Generate placebo treatment as random noise with same std as original treatment
        placebo_treatment = np.random.normal(loc=0, scale=np.std(original_treatment), size=len(df))

        # Initialize Causal Forest model with given seed for reproducibility
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            discrete_treatment=False,
            random_state=seed
        )
        # Fit model on outcome, placebo treatment, and normalized features
        model.fit(Y, placebo_treatment, X=X_norm)

        # Estimate Conditional Average Treatment Effects (CATE)
        cates = model.effect(X_norm)
        mean_cate = np.mean(cates)
        std_cate = np.std(cates)

        mean_cates.append(mean_cate)
        std_cates.append(std_cate)

    # Calculate mean and std deviation across seeds for the mean CATE
    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)

    # Perform one-sample t-test to check if mean CATE is significantly different from zero
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    print(f"Placebo Test (over {len(seeds)} seeds):")
    print(f"Average Mean CATE = {mean_of_means:.6f}")
    print(f"Std Deviation of Mean CATEs = {std_of_means:.6f}")
    print(f"T-Test: t = {t_stat:.3f}, p = {p_value:.3f}")
    print("Expectation: Mean CATE close to 0, p-value > 0.05 means no significant placebo effect.")

    return mean_cates, std_cates, t_stat, p_value

if __name__ == "__main__":
    seeds = range(10)
    placebo_test(seeds=seeds)
