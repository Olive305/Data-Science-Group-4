import pandas as pd
from utils import load_excel, basic_data_cleanup, write_excel
import os

#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
def searcher(search,df, start_row=0):
    """
    function that searches for the specified search term within a dataframe
    :param search:
    :param df:
    :param start_row:
    :return:
    """

    line = df.loc[start_row:, 0].astype(str).str.contains(search, na=False)
    if not line.any():
        return pd.DataFrame(), start_row
    line_index = line[line].index[0]

    start = line_index + 2
    end = start +2
    while end < len(df):
        if df.iloc[end].isnull().all():
            break
        end += 1

    df_block = df.iloc[start:end].reset_index(drop=True)
    return df_block, end

def df_cleanup(df, year=23):
    """
    Put the dataframe in a usable format

    Args:
        df (dataframe): dataframe that needs to be cleaned up
        
    Return:
        df (dataframe): cleaned dataframe
    """
    
    # Delete the first row from the dataframe
    df = df.iloc[1:].reset_index(drop=True)
    
    # Swap rows and columns
    df = df.transpose().reset_index(drop=True)
    
    # Set the value of the first cell to 'Krankenkasse'
    df.iloc[0, 0] = "Krankenkasse"
    
    # Delete columns where the first index (row) is empty
    df = df.loc[:, df.iloc[0].notna() & (df.iloc[0] != "")]
    
    # Delete rows where the first index (column) is empty
    df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0] != "")].reset_index(drop=True)
    
    # Set the first row as index
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    
    df = basic_data_cleanup(df)
    
    # Import the matching df
    df_mapping = load_excel("../data/matching_tabelle.xlsx")
    
    # Merge df with df_mapping using 'Krankenkasse' as the key
    df = df_mapping.merge(df, left_on=f"Name_dem_{year}", right_on="Krankenkasse", how="left")
    
    return df

def find_income(df_income):
    """
    Finds the household net income table in a given dataframe using the keyword that identifies it.
    :param df_income: DataFrame to search in
    :return: DataFrame with income data
    """
    search = "In welche Klasse ordnen Sie Ihr monatliches Haushaltsnettoeinkommen ein?"

    start_row = 0
    df_result, end = searcher(search, df_income)
    start_row = end + 1
    while True:
        df, end = searcher(search, df_income, start_row)
        if df.empty:
            break

        # Merge as columns (side by side) on 'Krankenkasse'
        df_result = pd.concat([df_result, df], axis=1)

        start_row = end + 1
    df_result.reset_index(drop=True, inplace=True)
    return df_result

def find_income_23():
    """
    Runs the find_income() function using the Kundenmonitor 2023 data.
    :return: DataFrame with household income data
    """
    df = load_excel('../data/Kundenmonitor_GKV_2023.xlsx', sheet_name="Band", header=None)
    df = find_income(df)
    df = df.dropna(axis=1, how='all')
    return df

def find_income_24():
    """
    Runs the find_income() function using the Kundenmonitor 2024 data.
    :return: DataFrame with household income data
    """
    df = load_excel('../data/Kundenmonitor_GKV_2024.xlsx', sheet_name="Band", header=None)
    df = find_income(df)
    df = df.dropna(axis=1, how='all')
    return df

def find_household_size(df_household):
    """
    Finds the household size table in a given dataframe using the provided search term.
    :param df_household: DataFrame to search in
    :return: DataFrame with household size data
    """
    search = "Wie viele Personen leben in Ihrem Haushalt, Sie selbst mit eingeschlossen?"

    start_row = 0
    df_result, end = searcher(search, df_household)
    start_row = end + 1
    while True:
        df, end = searcher(search, df_household, start_row)
        if df.empty:
            break

        # Merge as columns (side by side) on 'Krankenkasse'
        df_result = pd.concat([df_result, df], axis=1)

        start_row = end + 1
    df_result.reset_index(drop=True, inplace=True)
    return df_result

def find_household_size_23():
    """
    Runs the find_household_size() function using the Kundenmonitor 2023 data.
    :param search_term: The search term to locate the household size table
    :return: DataFrame with household size data
    """
    df = load_excel('../data/Kundenmonitor_GKV_2023.xlsx', sheet_name="Band", header=None)
    df = find_household_size(df)
    df = df.dropna(axis=1, how='all')
    return df

def find_household_size_24():
    """
    Runs the find_household_size() function using the Kundenmonitor 2024 data.
    :param search_term: The search term to locate the household size table
    :return: DataFrame with household size data
    """
    df = load_excel('../data/Kundenmonitor_GKV_2024.xlsx', sheet_name="Band", header=None)
    df = find_household_size(df)
    df = df.dropna(axis=1, how='all')
    return df

    
    
def calc_avg_income(df, df_haushalt, year):
    """
    caculate the average brutto income per person in a certain health insurance company
    """
    
    # average brutto income per month (source: https://de.statista.com/statistik/daten/studie/161355/umfrage/monatliche-bruttoloehne-und-bruttogehaelter-pro-kopf-in-deutschland/)
    average_24 = 3862
    average_23 = 3667
    
    average_people_per_house = 2
    
    # Merge df with df_haushalt on 'Name_fm'
    df = df.merge(df_haushalt, on='Name_fm', how='left')
    
    # for each insurance company calculate the median brutto income (of the whole household)
    income_values = []
    for idx, row in df.iterrows():
        
        try:
            complete_netto_income = (
            row["Unter 1.000 Euro"] * 500 +
            row["1.000 b.u. 1.500 Euro"] * 1250 +
            row["1.500 b.u. 2.000 Euro"] * 1750 +
            row["2.000 b.u. 2.500 Euro"] * 2250 +
            row["2.500 b.u. 3.000 Euro"] * 2750 +
            row["3.000 b.u. 4.000 Euro"] * 3500 +
            row["4.000 b.u. 5.000 Euro"] * 4500 +
            row["5.000 Euro und mehr"] * 5500
            )
            total_count = (
            row["Unter 1.000 Euro"] +
            row["1.000 b.u. 1.500 Euro"] +
            row["1.500 b.u. 2.000 Euro"] +
            row["2.000 b.u. 2.500 Euro"] +
            row["2.500 b.u. 3.000 Euro"] +
            row["3.000 b.u. 4.000 Euro"] +
            row["4.000 b.u. 5.000 Euro"] +
            row["5.000 Euro und mehr"]
            )
            avg_netto_income = complete_netto_income / total_count if total_count != 0 else 0
            # convert from netto to brutto (simply multiply with 1.4 for approximate values)
            income = avg_netto_income * 1.4

        except Exception:
            # Use average values if calculation fails
            income = average_24 if year == 24 else average_23
            
        try:
            complete_household_num = (
                row["1 Person"] +
                row["2 Personen"] * 2 + 
                row["3 Personen"] * 3 +
                row["4 Personen"] * 4 +
                row["5 Personen"] * 5 +
                row["5 Personen und mehr"] * 6
            )
            
            total_count = (
                row["1 Person"] +
                row["2 Personen"] + 
                row["3 Personen"] +
                row["4 Personen"] +
                row["5 Personen"] +
                row["5 Personen und mehr"]
            )
            
            avg_household_num = complete_household_num / total_count if total_count != 0 else 0
            
        except Exception:
            # use avg values
            avg_household_num = average_people_per_house
            
        income = income / avg_household_num
        
        income_values.append(income)
    df["income"] = income_values
    
    df = df[["Name_fm", "income"]]
    df["year"] = year + 2000
    
    return df

def get_full_income_df():
    df_income_23 = find_income_23()
    df_income_24 = find_income_24()
    
    df_haushalt_23 = find_household_size_23()
    df_haushalt_24 = find_household_size_24()
    
    df_income_23 = df_cleanup(df_income_23, 23)
    df_income_24 = df_cleanup(df_income_24, 24)
    df_haushalt_23 = df_cleanup(df_haushalt_23, 23)
    df_haushalt_24 = df_cleanup(df_haushalt_24, 24)
    
    df_income_23 = calc_avg_income(df_income_23, df_haushalt_23, 23)
    df_income_24 = calc_avg_income(df_income_24, df_haushalt_24, 24)
    
    df_income = pd.concat([df_income_23, df_income_24], ignore_index=True)
    
    #fill empty rows with default values
    avg_income_23 = 3667
    avg_income_24 = 3862
    
    df_income.loc[(df_income["year"] == 2023) & (df_income["income"].isna()), "income"] = avg_income_23
    df_income.loc[(df_income["year"] == 2024) & (df_income["income"].isna()), "income"] = avg_income_24
    
    write_excel(df_income, "../data/Brutto_income_23_24.xlsx")
    return df_income
    
if __name__ == "__main__":
    get_full_income_df()
    

"""
Zuerst wurde das monatliches Haushaltsnettoeinkommen aus der Kundenmonitor_GKV Tabelle geholt. Es wurde auch die durchschnittliche Nummer von Menschen pro Haushalt dort geholt.
Diese werte wurden nicht als durchschnittliche werte angegeben, sondern in ranges. Also wie viele angegeben haben wie viel zwischen 1000 und 1500 verdienen sozusagen.
Beides wurde dann für jede Krankenkasse in einen durchschnittswert umgerechnet
Dann wurde Nettoeinkommen durch Haushalt / Personen im Haushalt geteilt. (Mit der Annahme dass alle Personen eines Haushalts in der Selben krankenkasse versichert sind, wodurch sich Fehler durch Nichtverdiener im Haushalt ausgleichen)
Die Nettoeinkommen pro Person wurden mit 1.4 multipliziert um die Bruttoeinkommen abzuschätzen
Krankenkassen wo diese Werte nicht zur Verfügung standen, wurden die durchschnittlichen Bruttoeinkommen des jeweiligen Jahres (23, 24) in Deutschland verwendet (source: https://de.statista.com/statistik/daten/studie/161355/umfrage/monatliche-bruttoloehne-und-bruttogehaelter-pro-kopf-in-deutschland/)
"""