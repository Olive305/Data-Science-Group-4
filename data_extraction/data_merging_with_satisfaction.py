import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.data_merging import fuz_combine_fees_morbidity
from data_extraction.utils import write_excel, load_excel, basic_data_cleanup
from data_extraction.customer_data_extraction import process_excel_file

def kundenmonitor_extraction(year):

    if year != 2023 and year != 2024:
        print(f"Error: Only 2023 and 2024 are supported. Got year={year}")
        return None

    # Switch statement for file names based on year
    file_map = {
        2023: os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx'),
        2024: os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2024.xlsx')
    }

    file_name = file_map.get(year)
    if not file_name:
        print(f"Error: File name not found")
        return None

    process_excel_file(file_name, year)

    # Get the path for the custom summary file for the given year
    file_name = os.path.join(os.path.dirname(__file__), f'../data/custom_files/summary_df_{year}.xlsx')
    df = pd.read_excel(file_name)

    df.rename(columns={df.columns[0]: "Krankenkasse"}, inplace=True)
    
    df = basic_data_cleanup(df, 'Krankenkasse')
    
    # Store the cleaned DataFrame as an Excel file for testing
    test_output_dir = os.path.join(os.path.dirname(__file__), '../data/test_outputs')
    os.makedirs(test_output_dir, exist_ok=True)
    df.to_excel(os.path.join(test_output_dir, f'kundenmonitor_{year}_cleaned_test.xlsx'), index=False)

    # Set the first column as index, transpose, then reset index to preserve all data
    df_t = df.set_index(df.columns[0]).transpose().reset_index()

    df_t.rename(columns={df_t.columns[0]: "Krankenkasse"}, inplace=True)
    df_t = basic_data_cleanup(df_t, 'Krankenkasse')

    # Remove columns where any value is not in the range 1 to 5 (inclusive)
    cols_to_keep = [
        col for col in df_t.columns
        if df_t[col].dropna().apply(lambda x: 1 <= x <= 5 if isinstance(x, (int, float)) else True).all()
    ]
    df_t = df_t[cols_to_keep]

    return df_t

def merge_churn_with_satisfaction():

    try:
        full_data_path = os.path.join(os.path.dirname(__file__), '../data/full_data.xlsx')
        if os.path.exists(full_data_path):
            return pd.read_excel(full_data_path)
    except Exception as e:
        print(f"Warning: Could not read full_data.xlsx due to: {e}")

    # If file does not exist, continue as normal and save at the end

    # merge it with the matching table
    matching_path = os.path.join(os.path.dirname(__file__), '../data/matching_tabelle.xlsx')
    matching_df = pd.read_excel(matching_path)

    df_dem_24 = kundenmonitor_extraction(2024)
    df_dem_24.rename(columns={"Krankenkasse": "Name_dem_24"}, inplace=True)

    df_dem_23 = kundenmonitor_extraction(2023)
    df_dem_23.rename(columns={"Krankenkasse": "Name_dem_23"}, inplace=True)

    #renaming for merging
    matching_df.rename(columns={"Name_fm": "Krankenkasse"}, inplace=True)

    df_Kundenmonitor2023 = df_dem_23.merge(matching_df, on="Name_dem_23", how="left")
    df_Kundenmonitor2024 = df_dem_24.merge(matching_df, on="Name_dem_24", how="left")

    # Drop the 'Name_dem_23' and 'Name_dem_24' columns from the merged DataFrames
    df_Kundenmonitor2023_t = df_Kundenmonitor2023.drop(columns=['Name_dem_23'])
    df_Kundenmonitor2024_t = df_Kundenmonitor2024.drop(columns=['Name_dem_24'])

    # Add a 'Jahr' column to each DataFrame with the corresponding year
    df_Kundenmonitor2023_t['Jahr'] = 2023
    df_Kundenmonitor2024_t['Jahr'] = 2024

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

    # Align columns before concatenation to avoid InvalidIndexError
    common_cols = df_Kundenmonitor2023_t.columns.intersection(df_Kundenmonitor2024_t.columns)
    df_Kundenmonitor2023_t = df_Kundenmonitor2023_t[common_cols]
    df_Kundenmonitor2024_t = df_Kundenmonitor2024_t[common_cols]

    # Store both Kundenmonitor DataFrames as test Excel files
    test_output_dir = os.path.join(os.path.dirname(__file__), '../data/test_outputs')
    os.makedirs(test_output_dir, exist_ok=True)
    df_Kundenmonitor2023_t.to_excel(os.path.join(test_output_dir, 'kundenmonitor_2023_t_test.xlsx'), index=False)
    df_Kundenmonitor2024_t.to_excel(os.path.join(test_output_dir, 'kundenmonitor_2024_t_test.xlsx'), index=False)

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

    df_merged = df_merged.dropna(axis=1, how='all')

    # Remove columns with "unnamed" in the name (case-insensitive)
    df_merged = df_merged[[col for col in df_merged.columns if "unnamed" not in col.lower()]]

    df_merged.to_excel(os.path.join(os.path.dirname(__file__), '../data/full_data.xlsx'), index=False)

    return df_merged


if __name__ == "__main__":
    merge_churn_with_satisfaction()