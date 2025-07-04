import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import joblib
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.base import clone  # Import for cloning scikit-learn estimators

from data_extraction.utils import load_excel, normalize_features


def prepare_data(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
):
    """
    Prepares data for the Causal Forest model.
    Loads data, handles missing values, normalizes features,
    and sorts the DataFrame by the time period.
    """
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        # Perform data merging if the file is not found
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)

    # Fill missing numeric values with the median
    df = df.fillna(df.median(numeric_only=True))
    # Ensure the period column is in datetime format
    df[period_col] = pd.to_datetime(df[period_col])
    # Sort the entire DataFrame by the period column
    # This is crucial for sequential training and warm-starting
    df = df.sort_values(by=period_col)

    # Exclude columns that should not be used as features (X)
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X_full = df[[c for c in df.columns if c not in exclude]]
    # Only select numeric columns for X_full
    X_full = X_full.select_dtypes(include=[np.number])

    T_full = df[treatment_col].values
    Y_full = df[outcome_col].values

    # Return the full DataFrame and relevant arrays for later use
    return df, X_full, T_full, Y_full, period_col, treatment_col, outcome_col


def run_causal_forest_repeated_warm_start(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_warm_start.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    seeds=range(10),  # List of random seeds to use
):
    """
    Runs training of a Causal Forest model with warm-starting over periods.
    For each seed, starts a new model and warm-starts it over successive time periods.
    """
    # Prepare the data
    df_full, X_full_numeric, T_full, Y_full, period_col_name, t_col, y_col = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    # Get unique periods in chronological order
    unique_periods = df_full[period_col_name].unique()

    all_cates_per_seed_per_period = []  # Store CATEs per seed and period
    all_models_per_seed_per_period = []  # Store models per seed and period

    # Loop over each random seed
    for seed in seeds:
        print(f"\n--- Training for Seed: {seed} ---")
        current_model = None  # Initialize model for warm-start per seed
        cates_for_this_seed = []  # CATEs for the current seed
        models_for_this_seed = []  # Models for the current seed

        # Loop over each unique time period
        for i, period in enumerate(unique_periods):
            print(f"  Training up to period: {period.strftime('%Y-%m-%d')} (Cumulative)")

            # Select data up to the current period (inclusive)
            df_period = df_full[df_full[period_col_name] <= period].copy()

            # Prepare features (X), treatment (T), and outcome (Y) for the period
            exclude = {t_col, y_col, year_col, quarter_col, period_col_name}
            X_period = df_period[[c for c in df_period.columns if c not in exclude]]
            X_period = X_period.select_dtypes(include=[np.number])

            # Normalize features
            X_normalized_period, scaler_period = normalize_features(X_period)

            T_period = df_period[t_col].values
            Y_period = df_period[y_col].values

            if current_model is None:
                # On the first iteration (first period for this seed), initialize a new model
                model = CausalForestDML(
                    model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
                    model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
                    discrete_treatment=False,
                    random_state=seed
                )
            else:
                # For subsequent periods, warm-start the model with previous estimators
                model = CausalForestDML(
                    model_y=clone(current_model.model_y),  # Clone to get a fresh but pre-trained start
                    model_t=clone(current_model.model_t),  # Clone the treatment model
                    discrete_treatment=False,
                    random_state=seed
                )

            # Fit the model on the cumulative data
            model.fit(Y_period, T_period, X=X_normalized_period)
            current_model = model  # Save the fitted model for the next warm-start

            # Retrieve CATEs for the data of the current period
            cates_period = model.effect(X_normalized_period)
            cates_for_this_seed.append(cates_period)
            models_for_this_seed.append(model)

        all_cates_per_seed_per_period.append(cates_for_this_seed)
        all_models_per_seed_per_period.append(models_for_this_seed)

    # --- Aggregation and metric calculations ---
    # Use CATEs and models from the *last* period (after seeing all data) per seed
    final_period_cates_per_seed = [cates_list[-1] for cates_list in all_cates_per_seed_per_period]
    final_models_per_seed = [models_list[-1] for models_list in all_models_per_seed_per_period]

    # Average CATEs over seeds
    mean_cates = [np.mean(cates) for cates in final_period_cates_per_seed]
    # Standard deviation of CATE estimate intervals (mean uncertainty)
    std_cates = [np.mean(final_models_per_seed[s].effect_interval(normalize_features(X_full_numeric)[0])[1] - final_period_cates_per_seed[s])
                 for s in range(len(seeds))]

    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)  # Std of mean CATEs across seeds
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)  # T-test if mean CATE differs from zero

    # For robustness and heterogeneity metrics: all CATEs of final models on the *entire dataset*
    all_cates_array_final_models = np.vstack([model.effect(normalize_features(X_full_numeric)[0]) for model in final_models_per_seed]).T

    # Std of mean CATEs over observations (effect heterogeneity)
    mean_cates_per_obs = np.mean(all_cates_array_final_models, axis=1)
    std_of_mean_cates_over_obs = np.std(mean_cates_per_obs, ddof=1)

    # Mean std of CATEs over seeds per observation (robustness of estimate)
    std_cates_per_obs = np.std(all_cates_array_final_models, axis=1, ddof=1)
    mean_std_cates_across_obs = np.mean(std_cates_per_obs)

    # Policy value with the last model of the last seed on the entire dataset
    final_model = final_models_per_seed[-1]
    final_cates_full_data = final_model.effect(normalize_features(X_full_numeric)[0])
    policy = final_cates_full_data > 0  # Apply intervention if CATE is positive
    policy_value = np.mean(Y_full[policy])  # Expected outcome under this policy

    # KMeans clustering of final CATEs to identify subgroups
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    cate_clusters = kmeans.fit_predict(final_cates_full_data.reshape(-1, 1))

    # Scaler from the last normalization run on the full dataset
    _, final_scaler = normalize_features(X_full_numeric)

    # Bundle all key results and models
    model_bundle = {
        "model": final_model,
        "features": X_full_numeric.columns.tolist(),
        "scaler": final_scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": mean_of_means,
        "cate_std_mean": np.mean(std_cates),
        "feature_importance": final_model.feature_importances_,
        "mean_cates_all_seeds": mean_cates,
        "std_cates_all_seeds": std_cates,
        "t_stat": t_stat,
        "p_value": p_value,
        "policy_value": policy_value,
        "cate_clusters": cate_clusters,
    }

    # Save the model bundle
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"\nComplete model bundle saved at: {model_path}")
    print(f"Mean estimated treatment effect (CATE) across seeds (from last period): {mean_of_means:.6f}")
    print(f"Standard deviation of the mean CATEs across seeds: {std_of_means:.6f}")
    print(f"T-test against zero: t = {t_stat:.3f}, p = {p_value:.3f}")
    print(f"Standard deviation of the mean CATEs across observations (effect heterogeneity): {std_of_mean_cates_over_obs:.6f}")
    print(f"Mean standard deviation of CATE estimates across seeds/folds (robustness): {mean_std_cates_across_obs:.6f}")
    print(f"Policy value (expected outcome under model-based policy): {policy_value:.6f}")

    # Top 10 features for treatment effect heterogeneity (from the final model)
    importances = pd.Series(final_model.feature_importances_, index=X_full_numeric.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity (from the final model):")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle


def get_trained_causal_forest_warm_start():
    """Helper function to kick off warm-start Causal Forest training."""
    return run_causal_forest_repeated_warm_start()


if __name__ == "__main__":
    bundle = get_trained_causal_forest_warm_start()
    print("Warm-start Causal Forest model training completed.")
