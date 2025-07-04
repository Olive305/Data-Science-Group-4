import pandas as pd
import numpy as np
from econml.grf import CausalForest
import joblib
import os
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from data_extraction.utils import normalize_features, load_excel


def prepare_data(
        filepath="../data/fm_dem_sat_merged.xlsx",
        treatment_col="ZB_diff",
        outcome_col="Mitglieder_diff_next",
        year_col="Jahr",
        quarter_col="Quartal",
        period_col="Date",
):
    """
    Load dataset, impute missing values, normalize numeric features, extract treatment and outcome.

    Returns:
        df: Raw dataframe
        X: Original numeric features (non-normalized)
        X_normalized: Normalized feature matrix
        T: Treatment vector
        Y: Outcome vector
        scaler: Fitted normalization object
    """
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

    return df, X, X_normalized, T, Y, scaler


def select_features_with_double_lasso(X_normalized, T, Y, alpha=None):
    """
    Perform double selection Lasso for feature selection.

    Args:
        X_normalized: Normalized covariate matrix
        T: Treatment vector
        Y: Outcome vector
        alpha: Optional regularization strength (ignored when using LassoCV)

    Returns:
        Filtered feature matrix (DataFrame)
        List of selected feature names
    """
    LASSO_MAX_ITER = 100000

    lasso_y = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=LASSO_MAX_ITER)
    lasso_y.fit(X_normalized, Y)
    selected_features_y = X_normalized.columns[lasso_y.coef_ != 0].tolist()

    lasso_t = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=LASSO_MAX_ITER)
    lasso_t.fit(X_normalized, T)
    selected_features_t = X_normalized.columns[lasso_t.coef_ != 0].tolist()

    all_selected_features = list(set(selected_features_y) | set(selected_features_t))

    if not all_selected_features:
        print("Warning: Double Lasso selected no features. Using all features.")
        all_selected_features = X_normalized.columns.tolist()

    print(f"Number of original features: {X_normalized.shape[1]}")
    print(f"Number of selected features after Double Lasso: {len(all_selected_features)}")

    return X_normalized[all_selected_features], all_selected_features


def run_causal_forest_crossfit(
        filepath="../data/fm_dem_sat_merged.xlsx",
        model_path="../models/causal_forest_lasso.pkl",
        treatment_col="ZB_diff",
        outcome_col="Mitglieder_diff_next",
        year_col="Jahr",
        quarter_col="Quartal",
        period_col="Date",
        seeds=range(10),
        n_splits=3,
        perform_feature_selection=True,
):
    """
    Run cross-fitted honest Causal Forest with optional Double Lasso feature selection.

    Returns:
        Dictionary containing the trained model, metadata, and diagnostic statistics.
    """
    df, X_original, X_normalized_full, T, Y, scaler = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    # Optional feature selection using Double Lasso
    if perform_feature_selection:
        X_normalized, selected_features = select_features_with_double_lasso(X_normalized_full, T, Y)
        X_for_final_model = X_normalized
    else:
        X_normalized = X_normalized_full
        selected_features = X_original.columns.tolist()
        X_for_final_model = X_normalized

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
                n_estimators=200,
                min_samples_leaf=5,
                max_depth=15,
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
    st
