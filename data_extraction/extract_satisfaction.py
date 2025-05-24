import pandas as pd
import os
#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
def searcher(df, start_row=0):
    search = "Ausgewiesene Werte sind Mittelwerte: Alle Fragen auf einer fünfstufigen Skala von"
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
    location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2024.xlsx')
    df = pd.read_excel(location, sheet_name="Band", header=None)
    df_result= pd.DataFrame()
    start_row=0
    while True:
        df_block, end = searcher(df,start_row)
        if df_block.empty:
            break
        df_result = pd.concat([df_result, df_block], axis=1)
        start_row = end + 1
    df_result = df_result.drop(index=1).reset_index(drop=True)
    df_result.columns = df_result.iloc[0]
    df_result = df_result.drop(index=0).reset_index(drop=True)
    cols = list(df_result.columns)
    cols[0] = 'Attributes'
    df_result.columns = cols
    df_result = df_result.loc[:, df_result.columns.notna()]
    return df_result

#extract_satisfaction()