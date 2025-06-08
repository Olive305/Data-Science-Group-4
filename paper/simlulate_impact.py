from data_extraction.utils import column_name_cleanup, load_excel
import statsmodels.api as sm
import pandas as pd
import pickle
import numpy as np


def simulate_kk_impact(df, model_data, kk_name, adjustment_value):
    # Hole Daten für spezifische KK und Jahr 2024
    df = df[(df['Krankenkasse'] == kk_name) & (df['Jahr'] == 2024)].copy()
    df = column_name_cleanup(df)

    # Setze hypothetische Beitragsanpassung
    df['Zusatzbeitrag_diff'] = adjustment_value

    # Berechne Entity-Mittelwerte (wie in der Metaregression)
    df_mean = df.select_dtypes(include=[np.number]).mean().to_frame().T

    # Nur relevante Features
    X_raw = df_mean[model_data["feature_names"]].copy()

    # Standardisieren
    for col in model_data["feature_names"]:
        mean = model_data["means"][col]
        std = model_data["stds"][col]
        if std != 0:
            X_raw[col] = (X_raw[col] - mean) / std
        else:
            X_raw[col] = 0.0

    # Addiere Konstante (Intercept)
    X_std = sm.add_constant(X_raw)

    # Vorhersage: theta_i = geschätzte Sensitivität auf Zusatzbeitrag
    theta_hat = model_data["model"].predict(X_std).iloc[0]

    # Multipliziere mit hypothetischem Zusatzbeitrag_diff
    total_loss = theta_hat * adjustment_value

    return total_loss


def calc_impact():
    # Lade Daten
    try:
        df = load_excel('../data/fm_dem_sat_merged.xlsx')
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel('../data/fm_dem_sat_merged.xlsx')

    # Lade gespeichertes Modell + Metadaten
    with open("../models/metaregression_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Beispiel: Krankenkasse "AOK Bayern", Zusatzbeitragserhöhung um 0.1
    result = simulate_kk_impact(df, model_data, kk_name='aokbadenwürttemberg', adjustment_value=1)
    print(f"Geschätzter Mitgliederverlust bei 0.1 Beitragserhöhung: {result:.2f}")


if __name__ == '__main__':
    calc_impact()
