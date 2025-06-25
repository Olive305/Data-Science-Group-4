import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import joblib
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.base import clone # Import für das Klonen von Sci-Kit Learn Estimators

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
    Bereitet die Daten für das Causal Forest Modell vor.
    Lädt Daten, behandelt fehlende Werte, normalisiert Features
    und sortiert den DataFrame nach der Zeitperiode.
    """
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        # Führt die Datenzusammenführung aus, falls die Datei nicht gefunden wird
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)

    df = df.fillna(df.median(numeric_only=True))
    # Sicherstellen, dass die Periodenspalte im Datetime-Format ist
    df[period_col] = pd.to_datetime(df[period_col])
    # Den gesamten DataFrame nach der Periodenspalte sortieren.
    # Dies ist entscheidend für das sequentielle Training und Warm-Starting.
    df = df.sort_values(by=period_col)

    # Spalten ausschließen, die nicht als Features (X) verwendet werden sollen
    exclude = {treatment_col, outcome_col, year_col, quarter_col, period_col}
    X_full = df[[c for c in df.columns if c not in exclude]]
    # Nur numerische Spalten für X_full auswählen
    X_full = X_full.select_dtypes(include=[np.number])

    T_full = df[treatment_col].values
    Y_full = df[outcome_col].values

    # Rückgabe des vollständigen DataFrames und relevanter Spalten für die spätere Verwendung
    return df, X_full, T_full, Y_full, period_col, treatment_col, outcome_col


def run_causal_forest_repeated_warm_start(
    filepath="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_warm_start.pkl",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    seeds=range(10), # Liste der zu verwendenden Random Seeds
):
    """
    Führt das Training eines Causal Forest Modells mit Warm-Starting über Perioden durch.
    Für jeden Seed wird ein neues Modell gestartet, das dann über aufeinanderfolgende
    Zeitperioden warmgehalten wird.
    """
    # Daten vorbereiten
    df_full, X_full_numeric, T_full, Y_full, period_col_name, t_col, y_col = prepare_data(
        filepath=filepath,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        year_col=year_col,
        quarter_col=quarter_col,
        period_col=period_col,
    )

    # Eindeutige Perioden in chronologischer Reihenfolge erhalten.
    # Da df_full bereits sortiert ist, sollten diese auch sortiert sein.
    unique_periods = df_full[period_col_name].unique()

    all_cates_per_seed_per_period = [] # Speichert CATEs pro Seed und Periode
    all_models_per_seed_per_period = [] # Speichert Modelle pro Seed und Periode

    # Schleife über jeden Random Seed
    for seed in seeds:
        print(f"\n--- Training für Seed: {seed} ---")
        current_model = None # Modell für den Warm-Start pro Seed initialisieren
        cates_for_this_seed = [] # CATEs für den aktuellen Seed
        models_for_this_seed = [] # Modelle für den aktuellen Seed

        # Schleife über jede eindeutige Zeitperiode
        for i, period in enumerate(unique_periods):
            print(f"  Training bis Periode: {period.strftime('%Y-%m-%d')} (Kumulativ)")

            # Daten bis zur aktuellen Periode (einschließlich) auswählen
            df_period = df_full[df_full[period_col_name] <= period].copy()

            # Features (X), Treatment (T) und Outcome (Y) für die aktuelle Periode vorbereiten
            exclude = {t_col, y_col, year_col, quarter_col, period_col_name}
            X_period = df_period[[c for c in df_period.columns if c not in exclude]]
            X_period = X_period.select_dtypes(include=[np.number])

            # Features normalisieren
            X_normalized_period, scaler_period = normalize_features(X_period)

            T_period = df_period[t_col].values
            Y_period = df_period[y_col].values

            if current_model is None:
                # Beim ersten Durchlauf (erste Periode dieses Seeds) ein neues Modell initialisieren
                model = CausalForestDML(
                    model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
                    model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
                    discrete_treatment=False,
                    random_state=seed
                )
            else:
                # Für nachfolgende Perioden das Modell mit den Estimators des vorherigen Modells "warm" starten
                model = CausalForestDML(
                    model_y=clone(current_model.model_y), # Klonen, um einen frischen, aber vor-trainierten Start zu haben
                    model_t=clone(current_model.model_t), # Klonen des Treatment-Modells
                    discrete_treatment=False,
                    random_state=seed
                )

            # Modell auf den kumulativen Daten fitten
            model.fit(Y_period, T_period, X=X_normalized_period)
            current_model = model # Das aktuell gefittete Modell für den nächsten Warm-Start speichern

            # CATEs für die Daten der aktuellen Periode abrufen
            cates_period = model.effect(X_normalized_period)
            cates_for_this_seed.append(cates_period)
            models_for_this_seed.append(model)

        all_cates_per_seed_per_period.append(cates_for_this_seed)
        all_models_per_seed_per_period.append(models_for_this_seed)

    # --- Aggregation und Metriken-Berechnung ---
    # Die CATEs und Modelle der *letzten* Periode (nachdem alle Daten gesehen wurden) pro Seed verwenden
    final_period_cates_per_seed = [cates_list[-1] for cates_list in all_cates_per_seed_per_period]
    final_models_per_seed = [models_list[-1] for models_list in all_models_per_seed_per_period]

    # Durchschnittliche CATEs über die Seeds
    mean_cates = [np.mean(cates) for cates in final_period_cates_per_seed]
    # Standardabweichung der CATE-Schätzintervalle (mittlere Unsicherheit)
    std_cates = [np.mean(final_models_per_seed[s].effect_interval(normalize_features(X_full_numeric)[0])[1] - final_period_cates_per_seed[s])
                 for s in range(len(seeds))]

    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1) # Standardabweichung der mittleren CATEs über Seeds
    t_stat, p_value = stats.ttest_1samp(mean_cates, 0) # T-Test, ob der mittlere CATE signifikant von Null abweicht

    # Für Robustheit und Heterogenitätsmetriken: Alle CATEs der finalen Modelle auf den *gesamten Datensatz*
    all_cates_array_final_models = np.vstack([model.effect(normalize_features(X_full_numeric)[0]) for model in final_models_per_seed]).T

    # 1) Standardabweichung der mittleren CATEs über Beobachtungen (Effekt-Heterogenität)
    mean_cates_per_obs = np.mean(all_cates_array_final_models, axis=1)
    std_of_mean_cates_over_obs = np.std(mean_cates_per_obs, ddof=1)

    # 2) Mittlere Standardabweichung der CATEs über Seeds pro Beobachtung (Robustheit der Schätzung)
    std_cates_per_obs = np.std(all_cates_array_final_models, axis=1, ddof=1)
    mean_std_cates_across_obs = np.mean(std_cates_per_obs)

    # Policy Value mit dem letzten Modell des letzten Seeds auf den gesamten Datensatz
    final_model = final_models_per_seed[-1]
    final_cates_full_data = final_model.effect(normalize_features(X_full_numeric)[0])
    policy = final_cates_full_data > 0 # Wenn CATE positiv ist, die Intervention anwenden
    policy_value = np.mean(Y_full[policy]) # Erwartetes Outcome unter dieser Politik

    # KMeans Clustering der finalen CATEs zur Identifizierung von Untergruppen
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    cate_clusters = kmeans.fit_predict(final_cates_full_data.reshape(-1, 1))

    # Skalierer vom letzten Normalisierungsdurchlauf auf den vollständigen Datensatz
    _, final_scaler = normalize_features(X_full_numeric)


    # Alle wichtigen Ergebnisse und Modelle in einem Bundle speichern
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

    # Modell-Bundle speichern
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    print(f"\nKomplettes Modell-Bundle gespeichert unter: {model_path}")
    print(f"Mittlerer geschätzter Behandlungseffekt (CATE) über Seeds (aus letzter Periode): {mean_of_means:.6f}")
    print(f"Standardabweichung der mittleren CATEs über Seeds: {std_of_means:.6f}")
    print(f"T-Test gegen Null: t = {t_stat:.3f}, p = {p_value:.3f}")
    print(f"Standardabweichung der mittleren CATEs über Beobachtungen (Effekt-Heterogenität): {std_of_mean_cates_over_obs:.6f}")
    print(f"Mittlere Standardabweichung der CATE-Schätzungen über Seeds/Folds (Robustheit): {mean_std_cates_across_obs:.6f}")
    print(f"Policy Value (erwartetes Outcome bei Modell-basierter Politik): {policy_value:.6f}")


    # Top 10 Features nach Wichtigkeit für die Treatment-Effekt-Heterogenität
    importances = pd.Series(final_model.feature_importances_, index=X_full_numeric.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 Features für die Behandlungseffekt-Heterogenität (vom finalen Modell):")
    for feat, imp in top_features.items():
        print(f"{feat:30}: {imp:.4f}")

    return model_bundle


def get_trained_causal_forest_warm_start():
    """Hilfsfunktion zum Starten des Warm-Start Causal Forest Trainings."""
    return run_causal_forest_repeated_warm_start()


if __name__ == "__main__":
    bundle = get_trained_causal_forest_warm_start()
    print("Warm-Start Causal Forest Modelltraining abgeschlossen.")