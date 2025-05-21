from dataclasses import replace

import pandas as pd
import pandas as pd
from thefuzz import process, fuzz

def fuz_combine_fees_morbidity():
    #import data
    df_fees = pd.read_excel('../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
    df_morbidity = pd.read_excel('../data/Morbidity_Region.xlsx')

    df_morbidity['Krankenkasse'] = (
        df_morbidity['Krankenkasse']
        .replace('BKK der MTU Friedrichshafen', 'BKK MTU', regex=False)
        .replace('Hanseatische Krankenkasse (HEK)', 'HEK', regex=False)

    )




    #removing spaces and - writing in lowe case for easier matching
    df_fees['Krankenkasse'] = (
        df_fees['Krankenkasse']
        .str.lower()
        .str.replace('-', '', regex=True)
        .str.replace('–', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )

    df_morbidity['Krankenkasse'] = (
        df_morbidity['Krankenkasse']
        .str.lower()
        .str.replace('-', '', regex=True)
        .str.replace('–', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )

    # unique names from fees
    reference_names = df_fees['Krankenkasse'].unique()

    #fuzzy matching
    def match_name(name):
        match, score = process.extractOne(name, reference_names, scorer=fuzz.token_sort_ratio)
        return match if score >= 75 else name  # nur bei gutem Match ersetzen

    df_morbidity['Krankenkasse'] = df_morbidity['Krankenkasse'].apply(match_name)

    # outer merge -> keeps even the ones that are only availabe in 1 table
    df_merged = pd.merge(
        df_fees,
        df_morbidity,
        on=['Krankenkasse', 'Jahr'],
        how='outer',
        suffixes=('_fees', '_morbidity')
    )
    df_merged.to_excel("../data/merged_data.xlsx", index=False)
    return df_merged


fuz_combine_fees_morbidity()
"""
s1='metzingerbkk'
s2='bkkmetzinger'
print("fuzz score", fuzz.ratio(s1, s2))
"""
