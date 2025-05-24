import pandas as pd
import os
#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
def searcher(search,df, start_row=0):

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




def extract_satisfaction():
    """
    extracts the satisfaction from Kundenmonitor 2023 data
    return the df with the satisfaction
    """
    search = "Ausgewiesene Werte sind Mittelwerte: Alle Fragen auf einer fünfstufigen Skala von"
    location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2024.xlsx')
    df = pd.read_excel(location, sheet_name="Band", header=None)
    df_result= pd.DataFrame()
    start_row=0
    while True:
        df_block, end = searcher(search,df,start_row)
        if df_block.empty:
            break
        df_result = pd.concat([df_result, df_block], axis=1)
        start_row = end + 1
    df_result = df_result.drop(index=1).reset_index(drop=True)
    df_result.columns = df_result.iloc[0]
    df_result = df_result.drop(index=0).reset_index(drop=True)
    cols = list(df_result.columns)
    cols[0] = 'Krankenkasse'
    df_result.columns = cols
    df_result = df_result.loc[:, df_result.columns.notna()]
    df_result =df_result.T
    df_result.columns = df_result.iloc[0]
    df_result = df_result.drop(df_result.index[0])

    return df_result

def clean_demo(df):
    df = df.drop(df.columns[1], axis=1)  # drops Gesamt
    df = df.drop([2, 3]).reset_index(drop=True)  # drops n gesamt
    df = df.iloc[:-2].reset_index(drop=True)  # drops weiß nicht and summe
    prefixes = df.iloc[0]
    new_row = []
    current_prefix = None

    for i, val in enumerate(prefixes):
        if pd.notna(val):
            current_prefix = val
        combined = f"{current_prefix}_{df.iat[1, i]}"
        new_row.append(combined)

    df.iloc[1] = new_row

    df = df.drop(index=0).reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)
    cols = list(df.columns)
    cols[0] = 'Krankenkasse'
    df.columns = cols
    return df

def find_demographics():
    search = "Bei welcher gesetzlichen Krankenkasse sind Sie krankenversichert?"
    location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2024.xlsx')
    df_demo = pd.read_excel(location, sheet_name="Band", header=None)

    start_row=0
    df_result,end=searcher(search,df_demo)
    df_result = clean_demo(df_result)

    while True:
        df, end = searcher(search,df_demo,start_row)
        if df.empty:
            break

        df = clean_demo(df)
        df_result = pd.merge(df_result, df, on="Krankenkasse", how="left")

        start_row = end + 1

    return df_result


print(find_demographics())