import pandas as pd
import os
import matplotlib.pyplot as plt

def extract_xlsx_to_dataframe(file_path, sheet_name=0):
    """
    Extracts data from an Excel file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str or int, optional): Name or index of the sheet to extract. Defaults to the first sheet.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None
    

def visualize_data(df):
    """
    Visualizes the data in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to visualize.
    """
    if df is not None:
        krankenkasse = input("Enter the name of the \"Krankenkasse\" to visualize: ")
        filtered_df = df[df['Krankenkasse'] == krankenkasse]

        if not filtered_df.empty:
            # x axis: year
            # y axes: market share of insured and market share of members
            # This should add a plot for both
            ax = filtered_df.plot(kind='line', x='Jahr', y='Marktanteil Versicherte', color='blue', legend=True)
            filtered_df.plot(kind='line', x='Jahr', y='Marktanteil Mitglieder', color='red', secondary_y=True, ax=ax, legend=True)
            ax.set_ylabel('Marktanteil Versicherte')
            ax.right_ax.set_ylabel('Marktanteil Mitglieder')
            
            # Show the plot in a new window
            plt.show()
        else:
            print(f"No data found for Krankenkasse: {krankenkasse}")


if __name__ == "__main__":
    # Define the path to the Excel file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Marktanteile je Kasse.xlsx")
    
    # Extract data from the Excel file
    df = extract_xlsx_to_dataframe(file_path)
    
    # Visualize the data
    visualize_data(df)

