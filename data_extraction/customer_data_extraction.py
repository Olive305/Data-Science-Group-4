import pandas as pd

df_data = pd.read_excel('../data/Kundenmonitor_GKV_2024.xlsx', sheet_name='Band', header=None)
#read the excel

def find_subtables():
    subtables_amount = 0
    for idx,line in df_data[1].items():
        if line == "Gesamt":
            print(df_data.iloc[idx])
            subtables_amount += 1
    print(subtables_amount)

find_subtables()