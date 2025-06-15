import pandas as pd
import numpy as np
from econml.grf import CausalForest
import joblib
import os
from scipy import stats
from sklearn.linear_model import LassoCV  # Verwenden LassoCV für Kreuzvalidierung zur Auswahl von Alpha
from sklearn.preprocessing import StandardScaler
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
    X_normalized, scaler = normalize_features(X)  # Normalisierung VOR Lasso ist gut
    T = df[treatment_col].values
    Y = df[outcome_col].values

    return df, X, X_normalized, T, Y, scaler


# --- NEUE FUNKTION FÜR FEATURE SELECTION ---
def select_features_with_double_lasso(X_normalized, T, Y, alpha=None):
    """
    Führt Double Selection Lasso zur Feature-Auswahl durch.

    Args:
        X_normalized (pd.DataFrame): Normalisierte Kovariaten.
        T (np.array): Behandlungsvektor.
        Y (np.array): Outcome-Vektor.
        alpha (float, optional): Regularisierungsparameter für Lasso.
                                 Wenn None, wird LassoCV verwendet.

    Returns:
        pd.DataFrame: X mit den ausgewählten Features.
        list: Namen der ausgewählten Features.
    """

    # Höhere max_iter für LassoCV, um Konvergenzwarnungen zu vermeiden
    LASSO_MAX_ITER = 100000  # Erhöhter Wert

    # Lasso für Y
    lasso_y = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=LASSO_MAX_ITER)  # max_iter hier angepasst
    lasso_y.fit(X_normalized, Y)
    selected_features_y = X_normalized.columns[lasso_y.coef_ != 0].tolist()

    # Lasso für T
    lasso_t = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=LASSO_MAX_ITER)  # max_iter hier angepasst
    lasso_t.fit(X_normalized, T)
    selected_features_t = X_normalized.columns[lasso_t.coef_ != 0].tolist()

    # Vereinigung der Features
    all_selected_features = list(set(selected_features_y) | set(selected_features_t))

    # Sicherstellen, dass mindestens ein Feature ausgewählt wird, falls keine gefunden
    if not all_selected_features:
        print("Warnung: Double Lasso hat keine Features ausgewählt. Behalte alle Features.")
        all_selected_features = X_normalized.columns.tolist()

    print(f"Anzahl der ursprünglich Features: {X_normalized.shape[1]}")
    print(f"Anzahl der nach Double Lasso ausgewählten Features: {len(all_selected_features)}")

    return X_normalized[all_selected_features], all_selected_features


# --- ENDE DER NEUEN FUNKTION ---


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
        perform_feature_selection=True,  # Neuer Parameter
):
    df, X_original, X_normalized_full, T, Y, scaler = prepare_data(  # X_original für Spaltennamen
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    # --- HIER WIRD DIE FEATURE SELECTION EINGEFÜHRT ---
    if perform_feature_selection:
        X_normalized, selected_features = select_features_with_double_lasso(X_normalized_full, T, Y)
        # Aktualisiere die Original-Feature-Liste, falls nötig, z.B. für `feature_importance`
        X_for_final_model = X_normalized  # Für den finalen CausalForest
    else:
        X_normalized = X_normalized_full
        selected_features = X_original.columns.tolist()  # Alle Features, wenn keine Auswahl
        X_for_final_model = X_normalized
    # --- ENDE FEATURE SELECTION ---

    all_cates = []

    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_index, est_index in kf.split(X_normalized):  # X_normalized ist jetzt die reduzierte Menge
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
    std_of_means = mean_cates.std(ddof=1)
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    # Training des finalen Modells mit den ausgewählten Features
    final_model = CausalForest(
        n_estimators=200,
        min_samples_leaf=5,
        max_depth=15,
        honest=True,
        max_features="sqrt",
        random_state=seeds[-1],
    )
    final_model.fit(X_for_final_model, T.reshape(-1, 1), Y)  # Hier X_for_final_model verwenden

    # Calculate double robust scores (predicted effects)
    dr_scores = final_model.predict(X_for_final_model)  # Und hier
    policy = dr_scores > 0

    # Flatten arrays to 1D for comparison
    policy = policy.ravel()
    treatment_bool = (T > 0).ravel()

    if len(policy) != len(treatment_bool) or len(treatment_bool) != len(Y):
        raise ValueError("Missmatch of lengths")

    mask = (policy == treatment_bool)
    policy_value = np.mean(Y[mask])

    model_bundle = {
        "model": final_model,
        "features": selected_features,  # Die ausgewählten Features speichern
        "scaler": scaler,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "cate_mean": mean_of_means,
        "cate_std_mean": std_cates.mean(),
        "feature_importance": final_model.feature_importances_,
        # Feature Importance bezieht sich auf die TRAINIERTEN Features
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

    # Feature Importances basieren auf den ausgewählten Features
    importances = pd.Series(final_model.feature_importances_, index=selected_features)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features for treatment effect heterogeneity (honest):")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle


def get_trained_causal_forest():
    # Rufe die Funktion mit aktiviertem Feature Selection auf
    return run_causal_forest_crossfit(perform_feature_selection=True)


if __name__ == "__main__":
    bundle = get_trained_causal_forest()
    print("Honest causal forest training complete.")