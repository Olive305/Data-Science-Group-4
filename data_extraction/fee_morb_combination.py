import pandas as pd

df_fees = pd.read_excel('../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_morbidity = pd.read_excel('../data/Morbidity_Region.xlsx')

from thefuzz import process, fuzz

# Einmal alle einzigartigen Krankenkassen-Namen aus beiden DataFrames holen
names_fees = df_fees['Krankenkasse'].unique()
names_morbidity = df_morbidity['Krankenkasse'].unique()

# Mapping dict: Für jeden Namen in df_fees den besten Match in df_morbidity suchen
mapping = {}
for name in names_fees:
    best_match, score = process.extractOne(name, names_morbidity, scorer=fuzz.token_sort_ratio)
    mapping[name] = best_match

# Mapping auf df_fees anwenden, neue Spalte mit "normalisiertem" Namen
df_fees['Krankenkasse_norm'] = df_fees['Krankenkasse'].map(mapping)

# Jetzt die beiden DataFrames auf der "normalisierten" Krankenkasse und Jahr zusammenführen
df_merged = pd.merge(
    df_fees,
    df_morbidity,
    left_on=['Krankenkasse_norm', 'Jahr'],
    right_on=['Krankenkasse', 'Jahr'],
    how='left',
    suffixes=('_fees', '_morbidity')
)

# Optional: Die alte Spalte 'Krankenkasse' aus df_fees entfernen und die normierte Spalte umbenennen
df_merged.drop(columns=['Krankenkasse_fees'], inplace=True, errors='ignore')
df_merged.drop(columns=['Krankenkasse_morbidity'], inplace=True, errors='ignore')
df_merged.rename(columns={'Krankenkasse_norm': 'Krankenkasse'}, inplace=True)

