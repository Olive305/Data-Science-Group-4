import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity

# Get the df with the satisfaction values
file_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
df_sat = pd.read_excel(file_path)
df_sat.drop(['Churn_Rate_2023', 'Churn_Rate_2024'], axis=1, inplace=True)

# Switch the krankenkasse names to lowercase and remove spaces
df_sat['Krankenkasse'] = df_sat['Krankenkasse'].str.lower().str.replace(' ', '')

# Get the df with the morbidity rate and the churn rate
df_morb = fuz_combine_fees_morbidity()

# Sort both dataframes by 'Krankenkasse' and 'Jahr' for merge_asof
df_sat = df_sat.sort_values(['Jahr'])
df_morb = df_morb.sort_values(['Jahr'])

# Merge the dataframes using the year and the Krankenkasse values
# Since there are only satisfaciton values for 2023 and 2024, we use the nearest of the two years, when filling the table
df_merged = pd.merge_asof(
    df_morb,
    df_sat,
    on='Jahr',
    by='Krankenkasse',
    direction='nearest'
)

output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/full_data.xlsx')
df_merged.to_excel(output_path, index=False)



