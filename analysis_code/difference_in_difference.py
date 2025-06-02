from analysis_code.mixed_effects_model import mem
from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
from data_extraction.utils import load_excel

import statsmodels.formula.api as smf


def run_panel_regression(df, moderators=None):
    """
    Runs a panel regression with lagged treatment effect:
    Effect of ZB_diff_t on MG_diff_{t+1}.

    Parameters:
    - df: DataFrame with columns: 'KK' (Krankenkasse), 'Quartal', 'ZB_diff', 'MG_diff_next', plus ggf. Moderators
    - moderators: list of moderator variable names

    Returns:
    - fitted model result
    """

    # Baue Formel
    formula = "Mitglieder_diff_next ~ ZB_diff"
    if moderators:
        formula += " + " + " + ".join(moderators)

    # Füge Fixed Effects (Krankenkasse und Quartal) hinzu
    formula += " + C(Krankenkasse) + C(Date)"

    model = smf.ols(formula=formula, data=df).fit()
    return model

def panel():
    try:
        df = load_excel('../data/fm_dem_sat_merged.xlsx')
    except FileNotFoundError:
        merge_fm_dm_sat()
        df = load_excel('../data/fm_dem_sat_merged.xlsx')
    try:
        moderators = load_excel('../data/significant_moderators.xlsx')
    except FileNotFoundError:
        mem()
        moderators = load_excel('../data/significant_moderators.xlsx')
    mods = moderators['Moderator'].tolist()
    model = run_panel_regression(df, mods)
    return model
def run_did_regression(df, moderators=None,
                       dependent_var='Mitglieder_diff_next',
                       treatment_var='ZB_diff',
                       time_var='post'):
    formula = f"{dependent_var} ~ {treatment_var} + {time_var} + {treatment_var}:{time_var}"

    if moderators:
        formula += " + " + " + ".join(moderators)

    # Fit OLS regression
    model = smf.ols(formula=formula, data=df).fit()
    return model
def did():
    try:
        df = load_excel('../data/fm_dem_sat_merged.xlsx')
    except FileNotFoundError:
        merge_fm_dm_sat()
        df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df['post'] = (df['ZB_diff'] != 0).astype(int)
    moderators = load_excel('../data/significant_moderators.xlsx')
    mods = moderators['Moderator'].tolist()
    model = run_did_regression(df, mods)
    return model
#print(did().summary())