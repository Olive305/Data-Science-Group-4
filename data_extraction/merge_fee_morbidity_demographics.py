import pandas as pd

from analysis_code.predictive_models import regression_fm
from data_extraction.create_matching_table import build_matching_23
from data_extraction.data_extractor import find_demo_24, find_demo_23
from data_extraction.utils import load_excel, basic_data_cleanup, write_excel


def merge_fm_dem():
    """
    only works after running reg_morb_fee_churn()
    :return:
    """
    try:
        df_fm = load_excel("../data/prepared_regression_fm.xlsx")
    except FileNotFoundError:
        regression_fm()
        df_fm = load_excel("../data/prepared_regression_fm.xlsx")
    #print(df_fm.head())
    try:
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    except FileNotFoundError:
        build_matching_23()
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    df_dem_24 = find_demo_24()
    df_dem_24= basic_data_cleanup(df_dem_24,'Krankenkasse')
    df_dem_24.rename(columns={"Krankenkasse": "Name_dem_24"}, inplace=True)

    df_dem_23 = find_demo_23()
    df_dem_23 = basic_data_cleanup(df_dem_23,'Krankenkasse')
    df_dem_23.rename(columns={"Krankenkasse": "Name_dem_23"}, inplace=True)
    #renaming for merging
    df_mapping.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)

    df_fm = df_fm.merge(df_mapping, on="Krankenkasse", how="left")
    #adding the merged names to the df_fm
    #print(df_dem_24.head())

    df_fm_24 = df_fm[df_fm['Jahr']==2024].merge(df_dem_24, on="Name_dem_24", how="left")
    #print(df_fm_24.head())
    df_fm_23 = df_fm[df_fm['Jahr'] == 2023].merge(df_dem_23, on="Name_dem_23", how="left")
    
    df_combined = pd.concat([df_fm_24, df_fm_23], ignore_index=True)

    df_combined = df_combined.drop(columns=['Name_dem_24', 'Name_dem_23'])
    write_excel(df_combined, '../data/fm_dem_merged.xlsx')
    return df_combined
merge_fm_dem()