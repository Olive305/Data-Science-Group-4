import pandas as pd

from data_extraction.utils import load_excel

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


def clean_sat(df):
    """
    cleans the dataframe
    :param df:
    :return:
    """
    df = df.drop(index=1).reset_index(drop=True)
    df = df.drop(df.columns[1], axis=1)
    #df = df.drop(index=0).reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)
    cols = list(df.columns)
    cols[0] = 'Krankenkasse'
    df.columns = cols
    df = df.dropna(axis=1, how='all')
    return df


def extract_satisfaction(path, sheetname):
    """
    extracts the satisfaction from Kundenmonitor 2023 data
    return the df with the satisfaction
    """
    search = "Ausgewiesene Werte sind Mittelwerte: Alle Fragen auf einer fünfstufigen Skala von"

    df=load_excel(path, sheet_name=sheetname, header=None)
    df_result, end = searcher(search, df)
    df_result =clean_sat(df_result)
    #print(df_result)
    start_row=end +1
    while True:
        df_block, end = searcher(search,df,start_row)
        if df_block.empty:
            break
        df_block = clean_sat(df_block)
        df_block = df_block.drop(df_block.columns[0], axis=1)
        df_result = pd.concat([df_result, df_block], axis=1)
        #df_result = pd.merge(df_result, df_block, on="Krankenkasse", how="left")
        start_row = end + 1
    df_result = df_result.T
    df_result = df_result.reset_index()
    df_result.columns = df_result.iloc[0]
    df_result = df_result.drop(index=0).reset_index(drop=True)
    df_result = df_result.dropna(axis=1, how='all')
    df_result = df_result.loc[:, ~df_result.columns.duplicated(keep=False)]
    return df_result

def clean_demo(df):
    # Drop 'Gesamt' column and irrelevant rows
    df = df.drop(df.columns[1], axis=1)
    df = df.drop([2, 3]).reset_index(drop=True)
    df = df.iloc[:-2].reset_index(drop=True)

    # Generate combined column names from first two header rows
    prefixes = df.iloc[0]
    new_row = []
    current_prefix = None

    for i, val in enumerate(prefixes):
        if pd.notna(val):
            current_prefix = val
        suffix = str(df.iat[1, i]) if pd.notna(df.iat[1, i]) else f"unk_{i}"
        combined = f"{current_prefix}_{suffix}"
        new_row.append(combined)

    # Apply new column names
    df.iloc[1] = new_row
    df = df.drop(index=0).reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)

    # Rename first column to 'Krankenkasse'
    cols = list(df.columns)
    cols[0] = 'Krankenkasse'
    df.columns = cols

    # Drop columns that are entirely NaN or zero (invalid/unusable)
    df = df.dropna(axis=1, how='all')

    # --- Normalize grouped demographic columns ---
    from collections import defaultdict
    import numpy as np

    prefix_groups = defaultdict(list)
    for col in df.columns[1:]:
        prefix = col.split('_')[0]
        prefix_groups[prefix].append(col)

    for prefix, group_cols in prefix_groups.items():
        if len(group_cols) == 1:
            continue  # nothing to normalize

        sum_col = df[group_cols].sum(axis=1)
        sum_col = sum_col.astype(float).replace(0, np.nan)

        for col in group_cols:
            df[col] = df[col] / sum_col

        # Drop last column of group (e.g., residual bucket)
        df = df.drop(columns=[group_cols[-1]])

    return df




def find_demographics(df_demo):
    """
    finds the demographic table in a given dataframe using the keyword that identifies it
    :param df_demo:
    :return: df_demo
    """
    search = "Bei welcher gesetzlichen Krankenkasse sind Sie krankenversichert?"

    start_row=0
    df_result,end=searcher(search,df_demo)
    df_result = clean_demo(df_result)
    start_row = end + 1
    while True:
        df, end = searcher(search,df_demo,start_row)
        if df.empty:
            break

        df = clean_demo(df)
        df_result = pd.merge(df_result, df, on="Krankenkasse", how="left")

        start_row = end + 1
    df_result.reset_index(drop=True, inplace=True)
    return df_result

def find_demo_23():
    """
    runs the find_demographic() function using the Kundenmonitor 2023 data
    :return: df with the demographics
    """
    df = load_excel('../data/Kundenmonitor_GKV_2023.xlsx', sheet_name="Band", header = None)
    df = find_demographics(df)
    df = df.dropna(axis=1, how='all')
    return df
def find_demo_24():
    """
    runs the find_demographic() function using the Kundenmonitor 2024 data
    :return: df with the demographics
    """
    df = load_excel('../data/Kundenmonitor_GKV_2024.xlsx', sheet_name="Band", header = None)
    df = find_demographics(df)
    df = df.dropna(axis=1, how='all')
    return df
def checker():
    """
    this is legacy code
    but it compares the dfs with demographic data if they have matching KKs
    :return:
    """
    df1=find_demo_23()
    df2= find_demo_24()
    print(df1["Krankenkasse"].unique())
    print(df2["Krankenkasse"].unique())
    list1 = df1["Krankenkasse"].tolist()

    list2 = df2["Krankenkasse"].tolist()


    # XOR mit Sets
    xor = set(list1) ^ set(list2)
    print(xor)

def sat_23():
    """
    runs extract_satifsaction() using the data from 2023
    return the df with the satisfaction
    :return: df -> pd.DataFrame
    """
    df = extract_satisfaction('../data/Kundenmonitor_GKV_2023.xlsx', 'Band (2)')
    return df
def sat_24():
    """
    runs extract_satifsaction() using the data from 2024
    return the df with the satisfaction
    :return: df -> pd.DataFrame
    """
    df = extract_satisfaction('../data/Kundenmonitor_GKV_2024.xlsx', 'Band')
    return df
#print(find_demo_24())
#print(sat_24())