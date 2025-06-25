import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from data_extraction.utils import normalize_features, load_excel


def permutation_test_causal_forest(
    filepath,
    treatment_col,
    outcome_col,
    year_col,
    quarter_col,
    period_col,
    n_permutations=100,
    seed=42,
):
    np.random.seed(seed)
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
    X_norm, _ = normalize_features(X)

    Y = df[outcome_col].values
    T_original = df[treatment_col].values

    # Original CATE
    real_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
        discrete_treatment=False,
        random_state=seed,
    )
    real_model.fit(Y, T_original, X=X_norm)
    real_cate = real_model.effect(X_norm)
    real_mean_cate = np.mean(real_cate)

    # Nullverteilung
    null_distribution = []
    for i in range(n_permutations):
        T_perm = np.random.permutation(T_original)
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed + i),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed + i),
            discrete_treatment=False,
            random_state=seed + i,
        )
        model.fit(Y, T_perm, X=X_norm)
        perm_cate = model.effect(X_norm)
        null_distribution.append(np.mean(perm_cate))

    null_distribution = np.array(null_distribution)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(real_mean_cate))

    return real_mean_cate, null_distribution, p_value

# === Main Loop über viele Seeds ===

def multiple_seed_analysis(
    filepath="../data/fm_dem_sat_merged.xlsx",
    treatment_col="ZB_diff",
    outcome_col="Mitglieder_diff_next",
    year_col="Jahr",
    quarter_col="Quartal",
    period_col="Date",
    n_permutations=100,
    seeds=range(20),
):
    results = []

    for seed in seeds:
        print(f"Running seed {seed}")
        real_cate, null_dist, p_val = permutation_test_causal_forest(
            filepath=filepath,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            year_col=year_col,
            quarter_col=quarter_col,
            period_col=period_col,
            n_permutations=n_permutations,
            seed=seed,
        )
        results.append({
            "seed": seed,
            "real_mean_cate": real_cate,
            "null_mean_cate": np.mean(null_dist),
            "p_value": p_val,
        })

    df = pd.DataFrame(results)

    # Statistische Auswertung
    mean_cate = df["real_mean_cate"].mean()
    std_cate = df["real_mean_cate"].std()
    ci_lower = mean_cate - 1.96 * std_cate
    ci_upper = mean_cate + 1.96 * std_cate

    print("\n=== Zusammenfassung über alle Seeds ===")
    print(f"Durchschnittlicher Real-CATE: {mean_cate:.4f}")
    print(f"Standardabweichung: {std_cate:.4f}")
    print(f"Approx. 95%-Konfidenzintervall: [{ci_lower:.2f}, {ci_upper:.2f}]")
    signif_count = (df["p_value"] < 0.05).sum()
    print(f"Signifikante Seeds (p < 0.05): {signif_count} von {len(df)}")
    print("Durchschnittlicher P-Wert:", df["p_value"].mean())
    print("=======================================")

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(df["real_mean_cate"], bins=10, color='skyblue', edgecolor='black')
    axs[0].set_title("Verteilung der Real Mean CATEs")
    axs[0].axvline(mean_cate, color='red', linestyle='--', label='Mittelwert')
    axs[0].legend()

    axs[1].hist(df["p_value"], bins=10, color='salmon', edgecolor='black')
    axs[1].set_title("Verteilung der P-Werte")
    axs[1].axvline(0.05, color='black', linestyle='--', label='Signifikanzgrenze')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return df

# Ausführen
if __name__ == "__main__":
    df_results = multiple_seed_analysis()
