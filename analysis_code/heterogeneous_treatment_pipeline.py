import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from data_extraction.utils import load_excel, column_name_cleanup
import pickle
import os

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
                      treatment_var: str = 'Zusatzbeitrag_diff',
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
                          treatment_var: str = 'Zusatzbeitrag_diff') -> pd.DataFrame:
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
                       treatment_var: str = 'Zusatzbeitrag_diff',
                       outcome_var: str = 'Mitglieder_diff_next',
                       time_var: str = 'Date') -> sm.regression.linear_model.RegressionResultsWrapper:
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
    agg = agg.rename(columns={entity_var: entity_var})

    meta_df = pd.merge(slopes_df, agg, on=entity_var)

    ignore_cols = [entity_var, 'theta_i', outcome_var, 'ZB_diff', 'Jahr', 'Quartal']
    feature_cols = [col for col in meta_df.columns if col not in ignore_cols]

    for col in feature_cols:
        median = meta_df[col].replace([np.inf, -np.inf], np.nan).median()
        meta_df[col] = meta_df[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    for col in feature_cols:
        col_mean = meta_df[col].mean()
        col_std = meta_df[col].std(ddof=0)
        if col_std != 0:
            meta_df[col] = (meta_df[col] - col_mean) / col_std

    current_features = feature_cols[:]
    while True:
        X = sm.add_constant(meta_df[current_features])
        y = meta_df['theta_i']
        model = sm.OLS(y, X).fit()
        pvalues = model.pvalues.drop('const', errors='ignore')
        max_pval = pvalues.max()
        if max_pval > 0.05:
            worst_feature = pvalues.idxmax()
            current_features.remove(worst_feature)
        else:
            break

    X_final = sm.add_constant(meta_df[current_features])
    meta_res_final = sm.OLS(y, X_final).fit()
    return meta_res_final


def main():
    df_panel = load_data()
    mixed_res = fit_mixed_effects(df_panel)
    print("Mixed-Effects Model Results:")
    print(mixed_res.summary())

    slopes_df = extract_random_slopes(mixed_res)
    print("First 5 random slopes:")
    print(slopes_df.head())

    # Metaregression + Rückgriff auf Daten aus run_metaregression
    numeric = df_panel.select_dtypes(include=[np.number])
    agg = numeric.groupby(df_panel['Krankenkasse']).mean().reset_index()
    meta_df = pd.merge(slopes_df, agg, on='Krankenkasse')

    ignore_cols = ['Krankenkasse', 'theta_i', 'Mitglieder_diff_next', 'ZB_diff', 'Jahr', 'Quartal']
    feature_cols = [col for col in meta_df.columns if col not in ignore_cols]

    for col in feature_cols:
        median = meta_df[col].replace([np.inf, -np.inf], np.nan).median()
        meta_df[col] = meta_df[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    means = {}
    stds = {}
    for col in feature_cols:
        mean = meta_df[col].mean()
        std = meta_df[col].std(ddof=0)
        if std != 0:
            meta_df[col] = (meta_df[col] - mean) / std
            means[col] = mean
            stds[col] = std

    current_features = feature_cols[:]
    while True:
        X = sm.add_constant(meta_df[current_features])
        y = meta_df['theta_i']
        model = sm.OLS(y, X).fit()
        pvalues = model.pvalues.drop('const', errors='ignore')
        max_pval = pvalues.max()
        if max_pval > 0.05:
            worst_feature = pvalues.idxmax()
            current_features.remove(worst_feature)
        else:
            break

    X_final = sm.add_constant(meta_df[current_features])
    meta_res_final = sm.OLS(y, X_final).fit()

    print("Metaregression Results (Reduced Model):")
    print(meta_res_final.summary())

    model_data = {
        "model": meta_res_final,
        "feature_names": current_features,
        "means": {k: means.get(k, 0.0) for k in current_features},
        "stds": {k: stds.get(k, 1.0) for k in current_features},
    }


    with open("../models/metaregression_model.pkl", "wb") as f:
        pickle.dump(model_data, f)



if __name__ == '__main__':
    main()
