import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# import data
location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
df_Kundenmonitor2023 = pd.read_excel(location, sheet_name="EE")
location = os.path.join(os.path.dirname(__file__), '../data/custom_files/summary_df_2024.xlsx')
df_Kundenmonitor2024 = pd.read_excel(location)
location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_churn = pd.read_excel(location)

# Print heads for inspection
print(df_Kundenmonitor2023.head())
print(df_Kundenmonitor2024.head())

# Combine Year and Quarter for easier calculations
df_churn['Date'] = pd.to_datetime(df_churn['Jahr'].astype(str) + 'Q' + df_churn['Quartal'].astype(str))

# Calculate the percentage difference in members compared to the year after the current year
df_churn['Mitglieder_diff_next'] = (df_churn.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df_churn['Mitglieder']) / df_churn['Mitglieder']

# Calculate the average churn rate for each insurance company
average_churn = df_churn.groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()

# Create a new DataFrame with the average churn values
df_average_churn = average_churn.rename(columns={'Mitglieder_diff_next': 'Average_Churn_Rate'})

# Add a column to the dataframe with the churn rate in 2023 and 2024
df_churn_2023 = df_churn[df_churn['Jahr'] == 2023].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2023 = df_churn_2023.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2023'})
df_churn_2024 = df_churn[df_churn['Jahr'] == 2024].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2024 = df_churn_2024.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2024'})
df_average_churn = pd.merge(df_average_churn, df_churn_2023, on='Krankenkasse', how='left')
df_average_churn = pd.merge(df_average_churn, df_churn_2024, on='Krankenkasse', how='left')

print(df_average_churn.head)

# Prepare both Kundenmonitor datasets for merging
def prepare_kundenmonitor(df, year):
    df_t = df.set_index('Unnamed: 0').transpose()
    df_t = df_t.reset_index().rename(columns={'index': 'Krankenkasse'})
    df_t['Year'] = year
    # Deduplicate column names robustly
    if hasattr(pd.io.parsers, 'ParserBase'):
        df_t.columns = pd.io.parsers.ParserBase._maybe_dedup_names(list(df_t.columns))
    else:
        # Manual deduplication fallback
        def dedup_columns(cols):
            counts = {}
            new_cols = []
            for col in cols:
                if col not in counts:
                    counts[col] = 0
                    new_cols.append(col)
                else:
                    counts[col] += 1
                    new_cols.append(f"{col}.{counts[col]}")
            return new_cols
        df_t.columns = dedup_columns(list(df_t.columns))
    return df_t

df_Kundenmonitor2023_t = prepare_kundenmonitor(df_Kundenmonitor2023, 2023)
df_Kundenmonitor2024_t = prepare_kundenmonitor(df_Kundenmonitor2024, 2024)

# Align columns before concatenation to avoid InvalidIndexError
common_cols = df_Kundenmonitor2023_t.columns.intersection(df_Kundenmonitor2024_t.columns)
df_Kundenmonitor2023_t = df_Kundenmonitor2023_t[common_cols]
df_Kundenmonitor2024_t = df_Kundenmonitor2024_t[common_cols]

# Concatenate both years
df_kundenmonitor_all = pd.concat([df_Kundenmonitor2023_t, df_Kundenmonitor2024_t], ignore_index=True)
df_kundenmonitor_all = pd.merge(
    df_kundenmonitor_all,
    df_average_churn[['Krankenkasse', 'Churn_Rate_2023', 'Churn_Rate_2024']],
    on='Krankenkasse',
    how='left'
)

if False:
    # Store this as a excel file
    output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
    df_kundenmonitor_all.to_excel(output_path, index=False)


