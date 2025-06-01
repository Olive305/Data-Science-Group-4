import pandas as pd

from analysis_code.predictive_models import regression_fm
from data_extraction.create_matching_table import build_matching_23
from data_extraction.data_extractor import find_demo_24, find_demo_23, sat_23, sat_24
from data_extraction.utils import load_excel, basic_data_cleanup, write_excel

"""
Everything that isdone here could probably be done much smoother in 1 simplified function as it
is basically the same steps repeated 4 times => later
"""
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
    # print(df_fm.head())
    try:
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    except FileNotFoundError:
        build_matching_23()
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    df_dem_24 = find_demo_24()
    df_dem_24 = basic_data_cleanup(df_dem_24, 'Krankenkasse')
    df_dem_24.rename(columns={"Krankenkasse": "Name_dem_24"}, inplace=True)

    df_dem_23 = find_demo_23()
    df_dem_23 = basic_data_cleanup(df_dem_23, 'Krankenkasse')
    df_dem_23.rename(columns={"Krankenkasse": "Name_dem_23"}, inplace=True)
    # renaming for merging
    df_mapping.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)

    df_fm = df_fm.merge(df_mapping, on="Krankenkasse", how="left")
    # adding the merged names to the df_fm
    # print(df_dem_24.head())

    df_fm_24 = df_fm[df_fm['Jahr'] == 2024].merge(df_dem_24, on="Name_dem_24", how="left")
    # print(df_fm_24.head())
    df_fm_23 = df_fm[df_fm['Jahr'] == 2023].merge(df_dem_23, on="Name_dem_23", how="left")

    df_combined = pd.concat([df_fm_24, df_fm_23], ignore_index=True)

    df_combined = df_combined.drop(columns=['Name_dem_24', 'Name_dem_23'])
    write_excel(df_combined, '../data/fm_dem_merged.xlsx', index = False)
    return df_combined
def merge_fm_dm_sat():
    try:
        df = load_excel('../data/fm_dem_merged.xlsx')
    except FileNotFoundError:
        merge_fm_dem()
        df = load_excel('../data/fm_dem_merged.xlsx')
    try:
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    except FileNotFoundError:
        build_matching_23()
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    df_mapping.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)
    df = df.merge(df_mapping, on="Krankenkasse", how="left")

    df_23= sat_23()
    df_23.columns = df_23.columns.astype(str).str.strip()
    df_24= sat_24()
    df_24.columns = df_24.columns.astype(str).str.strip()

    df_24 = basic_data_cleanup(df_24, 'Krankenkasse')
    df_24.rename(columns={"Krankenkasse": "Name_dem_24"}, inplace=True)
    df_23 = basic_data_cleanup(df_23, 'Krankenkasse')
    df_23.rename(columns={"Krankenkasse": "Name_dem_23"}, inplace=True)
    df_sat_24 = df[df['Jahr'] == 2024].merge(df_24, on="Name_dem_24", how="left")

    df_sat_23 = df[df['Jahr'] == 2023].merge(df_23, on="Name_dem_23", how="left")
    df_combined = pd.concat([df_sat_24, df_sat_23], ignore_index=True)

    df_combined = df_combined.drop(columns=['Name_dem_24', 'Name_dem_23'])
    write_excel(df_combined, '../data/fm_dem_sat_merged.xlsx', index = False)

merge_fm_dm_sat()