import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from data_extraction.utils import normalize_features
from scipy import stats

def lead_effect_test(filepath="../data/fm_dem_sat_merged.xlsx", seeds=range(10)):
    df = pd.read_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))

    df = df.sort_values(by=['Krankenkasse', 'Date'])
    print(df)
    # calculate the amount of members lost compared to the year after the current year
    df['Mitglieder_diff_prev'] = df.groupby('Krankenkasse')['Mitglieder'].shift(+1) - df['Mitglieder']
    df = df.dropna(subset=['Mitglieder_diff_prev'])
    # ACHTUNG: Hier musst du sicherstellen, dass du eine Spalte für "previous outcome" hast,
    # z.B. "Mitglieder_diff_prev" - falls nicht, musst du sie vorher berechnen.
    # Alternativ kannst du mit Zeitverschiebung in deinem DataFrame arbeiten.

    outcome_col_prev = "Mitglieder_diff_prev"  # Beispielspalte mit Lead Outcome
    treatment_col = "ZB_diff"

    exclude = {treatment_col, outcome_col_prev, "Jahr", "Quartal", "Date", "Mitglieder_diff_next"}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])

    X_norm, scaler = normalize_features(X)
    Y_lead = df[outcome_col_prev].values
    T = df[treatment_col].values

    mean_cates = []
    std_cates = []

    for seed in seeds:
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
            discrete_treatment=False,
            random_state=seed
        )
        model.fit(Y_lead, T, X=X_norm)

        cates = model.effect(X_norm)
        mean_cate = np.mean(cates)
        std_cate = np.std(cates)

        mean_cates.append(mean_cate)
        std_cates.append(std_cate)

    mean_of_means = np.mean(mean_cates)
    std_of_means = np.std(mean_cates, ddof=1)

    t_stat, p_value = stats.ttest_1samp(mean_cates, 0)

    print(f"Lead Effect Test (over {len(seeds)} seeds):")
    print(f"Average Mean CATE = {mean_of_means:.6f}")
    print(f"Std Deviation of Mean CATEs = {std_of_means:.6f}")
    print(f"T-Test: t = {t_stat:.3f}, p = {p_value:.3f}")
    print("Erwartung: Mean CATE nahe 0, p-Wert > 0.05 zeigt keine Lead Effects.")

    return mean_cates, std_cates, t_stat, p_value

if __name__ == "__main__":
    seeds = range(10)
    lead_effect_test(seeds=seeds)
