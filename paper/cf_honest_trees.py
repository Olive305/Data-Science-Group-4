import pandas as pd
import numpy as np
from econml.grf import CausalForest
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

    return df, X, X_normalized, T, Y, scaler


def run_causal_forest_crossfit(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_full_honest.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    seeds=range(10),
    n_splits=3,
):
    from sklearn.model_selection import KFold

    df, X, X_normalized, T, Y, scaler = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    #kf = KFold(n_splits=n_splits, shuffle=True, random_state=seeds)
    all_cates = []

    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_index, est_index in kf.split(X_normalized):
            X_train, T_train, Y_train = (
                X_normalized.iloc[train_index],
                T[train_index],
                Y[train_index],
            )
            X_est = X_normalized.iloc[est_index]

            model = CausalForest(
                n_estimators=300,
                min_samples_leaf=10,
                max_depth=20,
                honest=True,
                max_features="sqrt",
                random_state=seed,
            )
            model.fit(X_train, T_train.reshape(-1, 1), Y_train)
            cates = model.predict(X_est)
            all_cates.append(cates.reshape(-1, 1))

    df_all = pd.DataFrame(np.hstack(all_cates))
    mean_cates = df_all.mean(axis=1)
    std_cates = df_all.std(axis=1)

    mean_of_means = mean_cates.mean()
    std_of_means = mean_cates.std(ddof=1)
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    final_model = CausalForest(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=10,
        honest=True,
        #max_features="sqrt",
        random_state=seeds[-1],
    )
    final_model.fit(X_normalized, T.reshape(-1, 1), Y)

    # Calculate double robust scores (predicted effects)
    dr_scores = final_model.predict(X_normalized)
    policy = dr_scores > 0

    # Flatten arrays to 1D for comparison
    policy = policy.ravel()
    treatment_bool = (T > 0).ravel()

    if len(policy) != len(treatment_bool) or len(treatment_bool) != len(Y):
        raise ValueError("Längen von policy, T und Y stimmen nicht überein.")

    mask = (policy == treatment_bool)
    policy_value = np.mean(Y[mask])

    model_bundle = {
        "model": final_model,
        "features": X.columns.tolist(),
        "scaler": scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": mean_of_means,
        "cate_std_mean": std_cates.mean(),
        "feature_importance": final_model.feature_importances_,
        "mean_cates_all": mean_cates,
        "std_cates_all": std_cates,
        "t_stat": t_stat,
        "p_value": p_value,
        "policy_value": policy_value,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"Honest causal forest model bundle saved to: {model_path}")
    print(f"Mean estimated treatment effect (CATE) averaged over folds and seeds: {mean_of_means:.6f}")
    print(f"Std deviation of mean CATEs over observations: {std_of_means:.6f}")
    print(f"T-Test against zero: t = {t_stat:.3f}, p = {p_value:.3f}")
    print(f"Mean standard deviation of CATE estimates across seeds/folds: {std_cates.mean():.6f}")
    print(f"Policy Value (expected outcome from model policy): {policy_value:.6f}")

    importances = pd.Series(final_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity (honest):")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle


def get_trained_causal_forest():
    return run_causal_forest_crossfit()


if __name__ == "__main__":
    bundle = get_trained_causal_forest()
    print("Honest causal forest training complete.")