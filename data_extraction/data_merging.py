import os
import pandas as pd
from thefuzz import process, fuzz

from data_extraction.data_extractor import find_demographics
from data_extraction.utils import data_cleanup

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
    df_fees['Krankenkasse'] = data_cleanup(df_morbidity['Krankenkasse'])
    df_morbidity['Krankenkasse'] = data_cleanup(df_morbidity['Krankenkasse'])

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


def merge_fm_dem():
    location = os.path.join(os.path.dirname(__file__), "../data/prepared_regression_fm.xlsx")
    df_fm = pd.read_excel(location)
    location = os.path.join(os.path.dirname(__file__), "../data/matching_tabelle.xlsx")
    df_mapping = pd.read_excel(location)
    df_dem = find_demographics()
    #renaming for merging
    df_mapping.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)

    #adding the merged names to the df_fm
    df_fm = df_fm.merge(df_mapping, on='Krankenkasse', how='left')

    df_dem.rename(columns={"Krankenkasse": "Name_dem"}, inplace=True)
    df_dem['Name_dem'] = data_cleanup(df_dem['Name_dem'])
    df_fm = df_fm.merge(df_dem, on='Name_dem', how='left')
    print(df_fm.head())



merge_fm_dem()

#fuz_combine_fees_morbidity()
"""
s1='metzingerbkk'
s2='bkkmetzinger'
print("fuzz score", fuzz.ratio(s1, s2))
"""
