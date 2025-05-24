import re
import unicodedata
import pandas as pd
import statsmodels.formula.api as smf


DATA_PATH = 'data/custom_files/full_data.xlsx'
df = pd.read_excel(DATA_PATH, engine='openpyxl')

def clean(col: str) -> str:
    # remove accents
    nfkd = unicodedata.normalize('NFKD', col)
    ascii_only = nfkd.encode('ascii', 'ignore').decode()
    # replace non-alphanumerics with underscore
    return re.sub(r'[^0-9A-Za-z]+', '_', ascii_only).strip('_')

df.columns = [clean(c) for c in df.columns]


df = df.dropna(subset=['Jahr', 'Quartal'])
df['Jahr'] = df['Jahr'].astype(int)
df['Quartal'] = df['Quartal'].astype(int)

# PeriodIndex & sort
df['Date'] = pd.PeriodIndex.from_fields(
    year=df['Jahr'], quarter=df['Quartal'], freq='Q'
)
df = df.sort_values(['Krankenkasse', 'Date']).reset_index(drop=True)

# Market share
df['Market_share'] = (
    df.groupby('Date')['Mitglieder']
      .transform(lambda x: x / x.sum() * 100)
)

# Next-quarter churn proxy
df['Mitglieder_pct_change_next'] = (
    df.groupby('Krankenkasse')['Mitglieder']
      .pct_change(periods=-1) * 100
)
df = df.dropna(subset=['Mitglieder_pct_change_next'])


df['fee_change_pp'] = (
    df.groupby('Krankenkasse')['Zusatzbeitrag']
      .pct_change() * 100
)
df['treated_bin'] = (df['fee_change_pp'] > 0).astype(int)
cutoff = pd.Period('2021Q1', freq='Q')
df['post'] = (df['Date'] >= cutoff).astype(int)
df['did']  = df['treated_bin'] * df['post']


desired_controls = [
    'Globalzufriedenheit_Schulnote',
    'Preis_Leistungs_Verhaeltnis_Schulnote',
    'Wiederwahlabsicht_Schulnote',
    'Kundenloyalitaet_Schulnote',
    'Risikofaktor',
    'Market_share'
]
# Keep only those present
controls = [c for c in desired_controls if c in df.columns]

# Coerce and drop
df[controls] = df[controls].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=controls + ['fee_change_pp'])


df['Krankenkasse'] = df['Krankenkasse'].astype(str).apply(clean)
df['fund_group'] = df['Krankenkasse']

df = pd.get_dummies(df, columns=['Krankenkasse','Date'], drop_first=True)
fe_terms = '+'.join([c for c in df.columns
                     if c.startswith('Krankenkasse_') or c.startswith('Date_')])
ctrl_str = '+'.join(controls)

# DID Model
formula_did = (
    'Mitglieder_pct_change_next ~ treated_bin + post + did'
    f' + {ctrl_str}'
    f' + {fe_terms}'
)
did_model = smf.ols(formula_did, data=df).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['fund_group']}

)
print("\n=== Difference-in-Differences ===")
print(did_model.summary())
