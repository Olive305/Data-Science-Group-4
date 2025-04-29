import pandas as pd
import os


def find_subtables(df_data):
    """
    Find and count all subtables in the excel file.

    Parameters:
        df (pd.DataFrame): DataFrame to analyze.
    """
    subtables_amount = 0
    for idx,line in df_data[1].items():
        if line == "Gesamt":
            #print(df_data.iloc[idx])
            subtables_amount += 1
    print("\nNumber of tables: ", subtables_amount)

# Function to find the end of the subtable
def find_subtable(df, start_row):
    """
    Find the end of a subtable in a DataFrame starting from a given row.
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        start_row (int): Row index to start searching for the end of the subtable
    Returns:
        pd.DataFrame: The subtable DataFrame
        int: The row index after the end of the subtable
    """

    empty_count = 0
    end_row = start_row
    for i in range(start_row, len(df)):
        if df.iloc[i].isnull().all():
            empty_count += 1
            if empty_count == 2:
                break
        else:
            empty_count = 0
        end_row = i
    # Slice the subtable from start_row to end_row (exclusive of the 2 empty rows)
    return df.iloc[start_row:end_row + 1], end_row + 1


def find_subtables_with_question(df_data):
    """
    Find and count all subtables in the excel file. Then the subtables are grouped, if they have the same question.

    Parameters:
        df (pd.DataFrame): DataFrame to analyze
    """
    
    subtables = []  # Store the subtables and their corresponding questions
    current_question_lines = []
    
    i = 0
    while i < len(df_data):
        # Store the first two lines as the current question lines
        if i < len(df_data) - 1:
            current_question_lines = df_data.iloc[i:i+2].values.flatten().tolist()
            i += 2
        else:
            break

        # Skip lines until the next "Gesamt" line
        while i < len(df_data) and df_data.iloc[i, 1] != "Gesamt":
            i += 1

        # If we reach a "Gesamt" line, we store the current table as a df
        if i < len(df_data) and df_data.iloc[i, 1] == "Gesamt":
            current_table, i = find_subtable(df_data, i)
            # Append the current table and question lines to the subtables list
            subtables.append([current_question_lines, current_table])
            current_question_lines = []  # Reset the current question lines

    # Group the subtables by their questions
    grouped_subtables = []
    last_question_lines = []
    for subtable in subtables:
        [question_lines, table] = subtable
        if question_lines == last_question_lines:
            grouped_subtables[-1][1] = pd.concat([grouped_subtables[-1][1], table.iloc[:, 2:]], ignore_index=True)
        else:
            grouped_subtables.append(subtable)
            last_question_lines = question_lines
        
    
    print("Subtables separated by 2-line gaps:")
    #for subtable in subtables:
    #    print(subtable)
    print(grouped_subtables[0])

    # Print the number of subtables
    print("\nNumber of subtables: ", len(grouped_subtables))


if __name__ == "__main__":

    # Read the excel file  
    file_path = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2024.xlsx')
    df_data = pd.read_excel(file_path, sheet_name='Band', header=None)
    
    # Find and count all subtables
    find_subtables_with_question(df_data)