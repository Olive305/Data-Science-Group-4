import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis_code.predictive_models import regression_fm
from data_extraction.create_matching_table import build_matching_23
from data_extraction.data_extractor import find_demo_24, find_demo_23, sat_23, sat_24
from data_extraction.utils import load_excel, basic_data_cleanup, write_excel

def merge_preperation():
    """
    import the table from regression preparation => merge of contribution increase and morbidity
    including the contribution change and Member churn
    import and merge the mapping ontop
    returns : df => pd.DataFrame
    """
    try:
        df = load_excel("../data/prepared_regression_fm.xlsx")
    except FileNotFoundError:
        regression_fm()
        df = load_excel("../data/prepared_regression_fm.xlsx")
    try:
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    except FileNotFoundError:
        build_matching_23()
        df_mapping = load_excel("../data/matching_tabelle.xlsx")
    #rename this column as it represents the name in the preparation_regression_fm
    df_mapping.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)
    df = df.merge(df_mapping, on="Krankenkasse", how ="left")
    return df

def merge_execution(df_result, df, name, jahr):
    """
    logic of merge
    cleans the Krankenkasse column for unified names
    renames the Krakenkasse column to the name given as an argument. This has to match the name from the matching table
    merges on that name and the year given as an argument
    :param df_result:
    :param df:
    :param name:
    :param jahr:
    :return: df => pd.DataFrame
    """
    df = basic_data_cleanup(df, 'Krankenkasse')
    df.rename(columns={"Krankenkasse": name}, inplace=True)
    df = df_result[df_result['Jahr'] == jahr].merge(df, on=name, how="left")
    return df

def merge_fm_dm_sat():
    """
    calls merge_preperation() to get the merged df of the data and name table
    Calls the functions that extract the information from the excel files
    Merges said information onto the afformentioned df
    :return: df => pd.DataFrame
    """
    df_result = merge_preperation()

    df_dem_24 = find_demo_24()
    df_dem_24 = merge_execution(df_result, df_dem_24, "Name_dem_24", 2024)

    df_dem_23 = find_demo_23()
    df_dem_23 = merge_execution(df_result, df_dem_23, "Name_dem_23",2023)

    df_result = pd.concat([df_dem_24, df_dem_23], ignore_index=True)
    df_result = df_result.drop(columns=['Name_dem_24', 'Name_dem_23'])

    df_sat_23 = sat_23()
    df_sat_23 = merge_execution(df_result, df_sat_23, "Name_sat_23",2023)

    df_sat_24 = sat_24()
    df_sat_24 = merge_execution(df_result, df_sat_24, "Name_sat_24",2024)

    df_result = pd.concat([df_sat_24, df_sat_23], ignore_index=True)

    df_result = df_result.drop(columns=['Name_sat_24', 'Name_sat_23'])
    df_result = df_result.dropna(axis=1, how='all')
    write_excel(df_result, '../data/fm_dem_sat_merged.xlsx', index=False)

if __name__ == "__main__":
    merge_fm_dm_sat()