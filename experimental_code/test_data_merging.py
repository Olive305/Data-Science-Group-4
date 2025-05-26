import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.data_merging import fuz_combine_fees_morbidity

if __name__ == "__main__":
    df_merged = fuz_combine_fees_morbidity()
    print(df_merged.head())  # Print the first few rows to check the output
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_files', 'test.xlsx')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_excel(output_path, index=False)
    print(f"DataFrame saved to {output_path}")