from data_extraction.utils import load_excel, column_name_cleanup
import statsmodels.formula.api as smf

def run_panel_regression(df, dependent_var='Mitglieder_diff_next', treatment_var='ZB_diff', entity_var='Krankenkasse', time_var='Quartal'):
    """
    Runs a fixed-effects panel regression with lagged treatment effect:
    Effect of treatment_var at time t on dependent_var at t+1.

    Automatically uses all other columns (except specified vars) as controls.
    """
    # Identify control variables: all columns excluding the main vars
    exclude = {dependent_var, treatment_var, entity_var, time_var}
    controls = [col for col in df.columns if col not in exclude]

    # Build formula
    parts = [treatment_var] + controls
    formula = f"{dependent_var} ~ " + " + ".join(parts)
    # Add fixed effects
    formula += f" + C({entity_var}) + C({time_var})"

    model = smf.ols(formula=formula, data=df).fit()
    return model

def run_did_regression(df, dependent_var='Mitglieder_diff_next', treatment_var='ZB_diff', entity_var='Krankenkasse', time_var='Quartal'):
    """
    Runs a staggered DiD regression, using binary treatment and post indicators.

    Uses all other columns (except specified vars) as controls.
    """
    # Create binary treatment and post indicators
    df = df.copy()
    df['treatment'] = (df[treatment_var] != 0).astype(int)
    # Determine first treatment quarter for each entity
    first_treat = df[df['treatment'] == 1].groupby(entity_var)[time_var].min()
    df = df.join(first_treat.rename('first_treat'), on=entity_var)
    df['post'] = (df[time_var] >= df['first_treat']).astype(int)

    # Identify control variables
    exclude = {dependent_var, treatment_var, entity_var, time_var, 'treatment', 'post', 'first_treat'}
    controls = [col for col in df.columns if col not in exclude]

    # Build formula
    formula = (f"{dependent_var} ~ treatment * post"
               + (" + " + " + ".join(controls) if controls else "")
               + f" + C({entity_var}) + C({time_var})")

    model = smf.ols(formula=formula, data=df).fit()
    return model

def panel():
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df= column_name_cleanup(df)
    model = run_panel_regression(df)
    return model

def did():
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df = column_name_cleanup(df)
    model = run_did_regression(df)
    return model

if __name__ == '__main__':
    panel_model = panel()
    print(panel_model.summary())
    """
    did_model = did()
    print(did_model.summary())
    """
