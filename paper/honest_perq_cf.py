import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from scipy import stats
from data_extraction.utils import normalize_features

# --- Utility function to evaluate test metrics ---
def evaluate_cate_distribution(cates):
    mean_cate = np.mean(cates)
    std_cate = np.std(cates)
    return mean_cate, std_cate

def evaluate_test_results(cates_list):
    mean_of_means = np.mean(cates_list)
    std_of_means = np.std(cates_list, ddof=1)
    t_stat, p_value = stats.ttest_1samp(cates_list, 0)
    return mean_of_means, std_of_means, t_stat, p_value

# --- Lead Effect Test ---
def lead_effect_test_with_model(df, model, X_full):
    df['Mitglieder_diff_prev'] = df.groupby('Krankenkasse')['Mitglieder'].shift(+1) - df['Mitglieder']
    df = df.dropna(subset=['Mitglieder_diff_prev'])

    Y_lead = df['Mitglieder_diff_prev'].values
    T = df['ZB_diff'].values

    # Passendes X aus X_full extrahieren
    X_filtered = X_full.loc[df.index]

    model.fit(Y_lead, T, X=X_filtered)
    cates = model.effect(X_filtered)

    mean_cate, std_cate = evaluate_cate_distribution(cates)
    return mean_cate, std_cate


# --- Placebo Test ---
def placebo_test_with_model(df, model, X_norm):
    Y = df['Mitglieder_diff_next'].values
    placebo_T = np.random.normal(loc=0, scale=np.std(df['ZB_diff']), size=len(df))

    model.fit(Y, placebo_T, X=X_norm)
    cates = model.effect(X_norm)

    mean_cate, std_cate = evaluate_cate_distribution(cates)
    return mean_cate, std_cate

# --- Sensitivity Analysis (simple example: effect size correlation with treatment strength) ---
def sensitivity_analysis(cates, treatment):
    correlation = np.corrcoef(cates, treatment)[0, 1]
    return correlation

# --- Main training and testing function ---
def run_causal_forest_pipeline(config):
    df = pd.read_excel(config["filepath"])
    df = df.fillna(df.median(numeric_only=True))
    df = df.sort_values(by=['Krankenkasse', 'Date'])

    exclude = {"ZB_diff", "Mitglieder_diff_next", "Jahr", "Quartal", "Date"}
    X = df[[c for c in df.columns if c not in exclude]]
    X = X.select_dtypes(include=[np.number])
    X_norm, scaler = normalize_features(X)
    Y = df['Mitglieder_diff_next'].values
    T = df['ZB_diff'].values

    mean_cates = []
    std_cates = []
    lead_test_results = []
    placebo_test_results = []
    sensitivity_results = []

    for seed in config["seeds"]:
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=seed),
            model_t=RandomForestRegressor(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=seed),
            discrete_treatment=False,
            n_estimators=config["n_estimators"],
            min_samples_leaf=config.get("min_samples_leaf", 10),
            max_depth=config.get("max_depth", 20),
            honest=config.get("honest", True),
            random_state=seed
        )

        model.fit(Y, T, X=X_norm)
        cates = model.effect(X_norm)
        mean_cate, std_cate = evaluate_cate_distribution(cates)
        mean_cates.append(mean_cate)
        std_cates.append(std_cate)

        if config.get("lead_test"):
            lead_mean, lead_std = lead_effect_test_with_model(df.copy(), model, X_norm)
            lead_test_results.append(lead_mean)

        if config.get("placebo_test"):
            placebo_mean, placebo_std = placebo_test_with_model(df.copy(), model, X_norm)
            placebo_test_results.append(placebo_mean)

        if config.get("sensitivity_analysis"):
            correlation = sensitivity_analysis(cates, T)
            sensitivity_results.append(correlation)

    results = {
        "mean_cate": np.mean(mean_cates),
        "std_cate": np.mean(std_cates),
        "lead_test": evaluate_test_results(lead_test_results) if lead_test_results else None,
        "placebo_test": evaluate_test_results(placebo_test_results) if placebo_test_results else None,
        "sensitivity_analysis": np.mean(sensitivity_results) if sensitivity_results else None
    }

    return results

if __name__ == "__main__":
    config = {
        "filepath": "../data/fm_dem_sat_merged.xlsx",
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_leaf": 20,
        "honest": True,
        "seeds": range(5),
        "lead_test": True,
        "placebo_test": True,
        "sensitivity_analysis": True
    }

    results = run_causal_forest_pipeline(config)
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
