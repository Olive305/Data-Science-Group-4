import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from econml.grf import CausalForest
import joblib
import os
import random

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

def tune_causal_forest(
    X, T, Y, seeds=range(5), n_iter=20, n_splits=3, model_path="../models/causal_forest_tuned.pkl"
):
    param_dist = {
        "n_estimators": [100, 200, 400, 600],
        "min_samples_leaf": [5, 10, 20],
        "max_depth": [10, 15, 20, None],
        "max_features": ["sqrt", "log2", None],
    }

    best_params_all_seeds = []
    best_scores_all_seeds = []

    for seed in seeds:
        print(f"Tuning seed {seed} ...")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        best_score = -np.inf
        best_params = None

        for _ in range(n_iter):
            params = {k: random.choice(v) for k, v in param_dist.items()}
            model = CausalForest(
                n_estimators=params["n_estimators"],
                min_samples_leaf=params["min_samples_leaf"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                honest=True,
                random_state=seed,
            )

            scores = []
            for train_idx, test_idx in cv.split(X):
                X_train, T_train, Y_train = X.iloc[train_idx], T[train_idx], Y[train_idx]
                X_test = X.iloc[test_idx]

                model.fit(X_train, T_train.reshape(-1, 1), Y_train)
                preds = model.predict(X_test)
                scores.append(np.var(preds))

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"Seed {seed} - Beste Params: {best_params}, Score: {best_score:.4f}")
        best_params_all_seeds.append(best_params)
        best_scores_all_seeds.append(best_score)

    best_index = np.argmax(best_scores_all_seeds)
    best_params_final = best_params_all_seeds[best_index]
    best_seed = list(seeds)[best_index]

    print(f"\nFinal ausgewählte Parameter (Seed {best_seed}): {best_params_final}")

    final_model = CausalForest(
        n_estimators=best_params_final["n_estimators"],
        min_samples_leaf=best_params_final["min_samples_leaf"],
        max_depth=best_params_final["max_depth"],
        max_features=best_params_final["max_features"],
        honest=True,
        random_state=best_seed,
    )
    final_model.fit(X, T.reshape(-1, 1), Y)

    model_bundle = {
        "model": final_model,
        "features": X.columns.tolist(),
        "treatment_col": "ZB_diff",
        "outcome_col": "Mitglieder_diff_next",
        "cate_variance_score": best_scores_all_seeds[best_index],
        "best_params": best_params_final,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"Bestes Modell gespeichert unter: {model_path}")

    return model_bundle

if __name__ == "__main__":
    # Daten laden & vorbereiten
    df, X, X_normalized, T, Y, scaler = prepare_data()

    # Tuning und Modelltraining
    bundle = tune_causal_forest(X_normalized, T, Y)

    print("Tuning und Training abgeschlossen.")
    print("Beste Parameter:", bundle["best_params"])
