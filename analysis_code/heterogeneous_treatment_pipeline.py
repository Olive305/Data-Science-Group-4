import os
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from data_extraction.utils import load_excel, column_name_cleanup


def load_data(panel_path: str = '../data/fm_dem_sat_merged.xlsx') -> pd.DataFrame:
    """
    Load the panel dataset and clean column names.

    Returns:
        df_panel: DataFrame containing all variables
    """
    try:
        df_panel = load_excel(panel_path)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df_panel = load_excel(panel_path)

    df_panel = column_name_cleanup(df_panel)
    return df_panel


def fit_mixed_effects(df: pd.DataFrame,
                      entity_var: str = 'Krankenkasse',
                      time_var: str = 'Date',
                      treatment_var: str = 'ZB_diff',
                      outcome_var: str = 'Mitglieder_diff_next'):
    """
    Fit a mixed-effects model with random intercepts and random slopes for the treatment variable.

    Model specification:
      Fixed effects: outcome ~ treatment + time indicators
      Random effects: random intercept and slope for each entity (Krankenkasse)

    Returns:
        fit: Fitted MixedLM result
    """
    fe_formula = f"{outcome_var} ~ {treatment_var} + C({time_var})"
    md = smf.mixedlm(
        fe_formula,
        df,
        groups=df[entity_var],
        re_formula=f"~ {treatment_var}"
    )
    fit = md.fit(reml=False)
    return fit


def extract_random_slopes(mixed_res,
                          entity_var: str = 'Krankenkasse',
                          treatment_var: str = 'ZB_diff') -> pd.DataFrame:
    """
    Extract random slope estimates for the treatment variable for each entity.

    Combine the global fixed-effect coefficient with each entity's random slope.

    Returns:
        slopes_df: DataFrame with columns ['Krankenkasse', 'theta_i']
    """
    global_beta = mixed_res.params[treatment_var]
    rand_eff = mixed_res.random_effects

    rows = []
    for entity, re_ser in rand_eff.items():
        slope_i = global_beta + re_ser.get(treatment_var, 0.0)
        rows.append({entity_var: entity, 'theta_i': slope_i})

    slopes_df = pd.DataFrame(rows)
    return slopes_df


def run_metaregression(slopes_df: pd.DataFrame,
                       df_panel: pd.DataFrame,
                       entity_var: str = 'Krankenkasse',
                       treatment_var: str = 'ZB_diff',
                       outcome_var: str = 'Mitglieder_diff_next',
                       time_var: str = 'Date'):
    """
    Run an OLS metaregression of theta_i on all numeric feature means.

    Steps:
    1. Aggregate numeric columns by entity, computing column means.
    2. Merge the random slopes (theta_i) with these aggregated features.
    3. Impute missing or infinite values with column medians.
    4. Standardize predictors to mean 0 and variance 1.
    5. Iteratively drop predictors with p > 0.05 and refit until only significant variables remain.

    Returns:
        meta_res_final: Fitted OLS regression result (reduced model)
    """
    numeric = df_panel.select_dtypes(include=[np.number])
    agg = numeric.groupby(df_panel[entity_var]).mean().reset_index()
    meta_df = pd.merge(slopes_df, agg, on=entity_var)

    ignore_cols = [entity_var, 'theta_i', outcome_var, treatment_var, 'Jahr', 'Quartal']
    feature_cols = [col for col in meta_df.columns if col not in ignore_cols]

    # Impute missing or infinite
    for col in feature_cols:
        median = meta_df[col].replace([np.inf, -np.inf], np.nan).median()
        meta_df[col] = meta_df[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    # Standardize
    means = {}
    stds = {}
    for col in feature_cols:
        m = meta_df[col].mean()
        s = meta_df[col].std(ddof=0)
        if s != 0:
            meta_df[col] = (meta_df[col] - m) / s
        means[col] = m
        stds[col] = s if s != 0 else 1.0

    # Iterative feature selection
    current_features = feature_cols[:]
    while True:
        X = sm.add_constant(meta_df[current_features])
        y = meta_df['theta_i']
        model = sm.OLS(y, X).fit()
        pvalues = model.pvalues.drop('const', errors='ignore')
        max_pval = pvalues.max()
        if max_pval > 0.05:
            worst = pvalues.idxmax()
            current_features.remove(worst)
        else:
            break

    # Final fit
    X_final = sm.add_constant(meta_df[current_features])
    meta_res_final = sm.OLS(meta_df['theta_i'], X_final).fit()
    return meta_res_final, current_features, means, stds


def slope_meta():
    df_panel = load_data()
    mixed_res = fit_mixed_effects(df_panel)
    print("Mixed-Effects Model Results:")
    print(mixed_res.summary())

    slopes_df = extract_random_slopes(mixed_res)
    print("First 5 random slopes:")
    print(slopes_df.head())

    meta_res_final, features, means, stds = run_metaregression(slopes_df, df_panel)
    print("Metaregression Results (Reduced Model):")
    print(meta_res_final.summary())

    # Save via joblib
    os.makedirs("../models", exist_ok=True)
    model_bundle = {
        "slopes": slopes_df.set_index("Krankenkasse")["theta_i"].to_dict(),
        "model": meta_res_final,
        "features": features,
        "means": means,
        "stds": stds,
        "add_constant": True
    }
    joblib.dump(model_bundle, "../models/metaregression_model.pkl")
    print("Meta-regression model saved to ../models/metaregression_model.pkl")

if __name__ == '__main__':
    slope_meta()
