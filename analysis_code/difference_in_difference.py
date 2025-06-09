import pandas as pd
import statsmodels.formula.api as smf

from data_extraction.utils import load_excel, column_name_cleanup


def run_stratified_regression(df, dependent_var='Mitglieder_diff_next', treatment_var='ZB_diff', time_var='Date'):
    """
    Runs one regression across all data, but comparisons are only made within the same Date.
    Date is included as a fixed effect to block comparisons across time.
    """

    # Convert to datetime
    df[time_var] = pd.to_datetime(df[time_var], errors='coerce')

    # Fill all numeric columns with median (including treatment & target)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Select control variables (exclude irrelevant or non-numeric)
    exclude = {
        dependent_var,
        treatment_var,
        'Krankenkasse',
        'Jahr',
        'Quartal',
        'first_treat',
        'post'
    }
    controls = [
        col for col in df.columns
        if col not in exclude and df[col].dtype.name in ['float64', 'int64']
    ]

    # Build regression formula
    parts = [treatment_var] + controls
    formula = f"{dependent_var} ~ " + " + ".join(parts)
    formula += f" + C({time_var})"  # fixed effect to compare only within same Date

    # Fit the model
    model = smf.ols(formula=formula, data=df).fit()
    return model


def run_stratified_did(df, dependent_var='Mitglieder_diff_next', treatment_var='ZB_diff', time_var='Date'):
    """
    Runs a DiD-style regression, stratified by Date.
    Treatment = binary (ZB_diff != 0), only compared within same Date.
    """

    df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
    df['treatment'] = (df[treatment_var] != 0).astype(int)

    # Determine first treatment time per entity
    first_treat = df[df['treatment'] == 1].groupby('Krankenkasse')[time_var].min()
    df = df.join(first_treat.rename('first_treat'), on='Krankenkasse')
    df['post'] = (df[time_var] >= df['first_treat']).astype(int)

    # Fill missing numerics with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Select control variables
    exclude = {
        dependent_var,
        treatment_var,
        'Krankenkasse',
        'Jahr',
        'Quartal',
        'first_treat',
        'post',
        'treatment'
    }
    controls = [
        col for col in df.columns
        if col not in exclude and df[col].dtype.name in ['float64', 'int64']
    ]

    # Build formula
    formula = f"{dependent_var} ~ treatment * post"
    if controls:
        formula += " + " + " + ".join(controls)
   # formula += f" + C({time_var})"  # restrict comparison to same Date

    model = smf.ols(formula=formula, data=df).fit()
    return model


def panel():
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df = column_name_cleanup(df)
    model = run_stratified_regression(df)
    return model


def did():
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df = column_name_cleanup(df)
    model = run_stratified_did(df)
    return model


if __name__ == '__main__':
    # Run DiD model stratified by Date
    did_model = did()
    print(did_model.summary())

    # Optionally: run the standard regression
    # panel_model = panel()
    # print(panel_model.summary())
