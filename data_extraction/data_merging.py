import pandas as pd
from thefuzz import process, fuzz

from data_extraction.utils import basic_data_cleanup, load_excel, write_excel

#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def fuz_combine_fees_morbidity():
    #import data
    df_fees = load_excel('../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
    df_morbidity = load_excel('../data/Morbidity_Region.xlsx')

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
    df_fees= basic_data_cleanup(df_fees,'Krankenkasse')
    df_morbidity = basic_data_cleanup(df_morbidity,'Krankenkasse')

    #print(df_morbidity)
    #print(df_fees)

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
    write_excel(df_merged, "../data/morb_fee_merged.xlsx", index = False)
#fuz_combine_fees_morbidity()
"""
s1='metzingerbkk'
s2='bkkmetzinger'
print("fuzz score", fuzz.ratio(s1, s2))
"""
