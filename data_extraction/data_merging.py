from dataclasses import replace
import os
import pandas as pd
from thefuzz import process, fuzz

#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def fuz_combine_fees_morbidity():
    #import data
    import os

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    fees_path = os.path.join(data_dir, 'Zusatzbeitrag_je Kasse je Quartal.xlsx')
    morbidity_path = os.path.join(data_dir, 'Morbidity_Region.xlsx')



    df_fees = pd.read_excel(fees_path, engine='openpyxl')
    df_morbidity = pd.read_excel(morbidity_path, engine='openpyxl')

    df_morbidity['Krankenkasse'] = (
        df_morbidity['Krankenkasse']
        .replace('BKK der MTU Friedrichshafen', 'BKK MTU', regex=False)
        .replace('Hanseatische Krankenkasse (HEK)', 'HEK', regex=False)
        .replace('BKK Metzinger','Metzinger BKK', regex=False)
    )
    #this would be combined with bkk24 from the other list but the BKK itself is not listed there
    df_morbidity = df_morbidity[df_morbidity['Krankenkasse'] != 'BKK']
    #for some reason it's listed twice for 2021
    df_morbidity = df_morbidity[
        ~((df_morbidity['Krankenkasse'] == 'IKK - Die Innovationskasse') & (df_morbidity['Risikofaktor'] == '-'))]

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
        return match if score >= 100 else name  # nur bei gutem Match ersetzen

    df_morbidity['Krankenkasse'] = df_morbidity['Krankenkasse'].apply(match_name)

    # outer merge -> keeps even the ones that are only availabe in 1 table
    df_merged = pd.merge(
        df_fees,
        df_morbidity,
        on=['Krankenkasse', 'Jahr'],
        how='outer',
        suffixes=('_fees', '_morbidity')
    )
    #print("länge merged", len(df_merged))
    #merged_path = os.path.join(data_dir, 'merged_data.xlsx')
    #df_merged.to_excel(merged_path, index=False)

    #print(df_merged[df_merged.duplicated(subset=['Krankenkasse','Jahr'])])
    return df_merged
def add_satisfation():
    location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
    df_Kundenmonitor2023 = pd.read_excel(location, sheet_name="EE")
    location = os.path.join(os.path.dirname(__file__), '../data/custom_files/summary_df_2024.xlsx')
    df_Kundenmonitor2024 = pd.read_excel(location)
    df_Kundenmonitor2023 = df_Kundenmonitor2023.T
    print(df_Kundenmonitor2023)


add_satisfation()
#fuz_combine_fees_morbidity()
"""
s1='metzingerbkk'
s2='bkkmetzinger'
print("fuzz score", fuzz.ratio(s1, s2))
"""
