import os
import sys
import pandas as pd

def merge_churn_with_satisfaction():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_extraction.data_merging import fuz_combine_fees_morbidity

    # import satisfaction data
    location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
    df_Kundenmonitor2023 = pd.read_excel(location, sheet_name="EE")
    location = os.path.join(os.path.dirname(__file__), '../data/custom_files/summary_df_2024.xlsx')
    df_Kundenmonitor2024 = pd.read_excel(location)

    # import other data
    fuz_combine_fees_morbidity()

    # Import the prepared_regression_fm.xlsx file
    morb_fee_path = os.path.join(os.path.dirname(__file__), '../data/prepared_regression_fm.xlsx')
    df_churn = pd.read_excel(morb_fee_path)

    # Fill empty Quartal values with 1
    df_churn['Quartal'] = df_churn['Quartal'].fillna(1)

    # Combine Year and Quarter for easier calculations
    df_churn['Quartal'] = df_churn['Quartal'].astype(int)
    df_churn['Date'] = pd.PeriodIndex(df_churn['Jahr'].astype(str) + 'Q' + df_churn['Quartal'].astype(str), freq='Q').to_timestamp()

    # Prepare both Kundenmonitor datasets for merging
    def prepare_kundenmonitor(df, year):
        df_t = df.set_index('Unnamed: 0').transpose()
        df_t = df_t.reset_index().rename(columns={'index': 'Krankenkasse'})
        df_t['Jahr'] = year
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

    # Switch the krankenkasse names to lowercase and remove spaces
    df_kundenmonitor_all['Krankenkasse'] = df_kundenmonitor_all['Krankenkasse'].str.lower().str.replace(' ', '')

    # Sort both dataframes by 'Krankenkasse' and 'Jahr' for merge_asof
    df_kundenmonitor_all = df_kundenmonitor_all.sort_values(['Jahr'])
    df_churn = df_churn.sort_values(['Jahr'])

    # Merge the dataframes using the year and the Krankenkasse values
    # Since there are only satisfaction values for 2023 and 2024, we use the nearest of the two years, when filling the table
    df_merged = pd.merge_asof(
        df_churn,
        df_kundenmonitor_all,
        on='Jahr',
        by='Krankenkasse',
        direction='nearest'
    )

    # Fill empty values of the df_merged with the mean of the column
    df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

    # Sort by 'Krankenkasse', 'Jahr', 'Quartal'
    df_merged = df_merged.sort_values(['Krankenkasse', 'Jahr', 'Quartal']).reset_index(drop=True)

    # Calculate percentual change in members compared to next quarter for each Krankenkasse
    df_merged['Mitglieder_pct_change_next'] = df_merged['Mitglieder_diff_next'] / df_merged['Mitglieder']
    
    return df_merged
