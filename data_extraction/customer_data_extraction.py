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


def find_subtable(df, start_row):
    """
    Find the end of a subtable in a DataFrame starting from a given row and then return it.
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
    Find and count all subtables in the excel file.
    All subtables with the same question are grouped together.


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
            # Remove all NaN values from the question lines
            current_question_lines = [line for line in current_question_lines if pd.notna(line)]
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

    for question_lines, table in subtables:
        if last_question_lines == question_lines:
            # Reset index to default integer index to avoid misalignment
            table = table.reset_index(drop=True)
            grouped_subtables[-1][1] = pd.concat([
                grouped_subtables[-1][1].reset_index(drop=True),
                table
            ], axis=1)
        else:
            grouped_subtables.append([question_lines, table])
            last_question_lines = question_lines


    if False:
        for subtable in subtables:
            print(subtable)

    print("Subtables separated by 2-line gaps:")
    #for subtable in subtables:
    #    print(subtable)
    print(grouped_subtables[0])

    # Print all the questions
    if False:
        print("\nQuestions:")
        for subtable in grouped_subtables:
            question_lines = subtable[0]
            print(question_lines, "\n\n")

    # Print the number of subtables
    print("\nNumber of subtables: ", len(grouped_subtables))

    return grouped_subtables

def analyze_subtable(grouped_subtables, number=4, df=None):
    """
    Analyze the satisfaction of the customers based on the grouped subtables.
    We use the subtable with index 4 (Globalzufriedenheit = Global satisfaction) for this analysis.

    Parameters:
        grouped_subtables (list): List of grouped subtables
        number (int): number of the subtable to analyze
        df (dataframe): dataframe to add the values
    """
    print(grouped_subtables[number])
    question, table = grouped_subtables[number]

    # Dynamically locate the columns based on the values in the first row
    selected_columns = ["Gesetzliche Krankenkasse", "Vergleich Kassensysteme", "AOK", "BKK", "BKK (Fortsetzung)", "IKK"]

    selected_columns_dict = {}

    column_indices = []
    seen_columns = set()
    for i, col in enumerate(table.iloc[0].values):
        if col in selected_columns and col not in seen_columns:
            column_indices.append(i)
            seen_columns.add(col)
            selected_columns_dict[i] = col

    for i in range(len(column_indices)):
        # starting from this column, check if the next column is empty
        # while the next column has nan values, add it to the column_indices
        df_index = column_indices[i]
        while df_index + 1 < len(table.columns) and pd.isna(table.iloc[0, df_index + 1]):
            column_indices.append(df_index + 1)
            df_index += 1
            selected_columns_dict[df_index] = selected_columns[i]

    # Remove duplicates and sort the column indices
    column_indices = sorted(set(column_indices))

    # Extract the Mittelwert values for the selected columns
    satisfaction_values = table.iloc[15, column_indices].values

    company_names = []

    # Extract the health insurance company names from the second row
    for column_index in column_indices:
        company_names.append(selected_columns_dict[column_index] + " - " + str(table.iloc[1, column_index]))

    # Combine the company names and satisfaction values, then sort by satisfaction
    # Filter out instances where satisfaction_values is not a number
    valid_entries = [(name, value) for name, value in zip(company_names, satisfaction_values) if pd.notna(value) and isinstance(value, (int, float))]

    # Sort the valid entries by satisfaction values in descending order
    sorted_companies = sorted(valid_entries, key=lambda x: x[1], reverse=True)


    print("\nHealth insurance companies sorted by satisfaction:")
    print(sorted_companies)

    # if there are no valid entries, skip adding to the DataFrame
    if not valid_entries:
        print(f"No valid entries found for subtable {number}. Skipping...")
        return df

    # if a dataframe was passed as an parameter, then add the values as a line
    if df is not None:
        df.loc[" ".join(question[0])] = {name: value for name, value in sorted_companies}
        return df
    else:
        # Create a new DataFrame with the values that would be added
        df = pd.DataFrame(
            {name: [value] for name, value in sorted_companies},
            index=["".join(question[0])]
        )
        return df



if __name__ == "__main__":

    # Read the excel file  
    file_path = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
    df_data = pd.read_excel(file_path, sheet_name='Band', header=None)
    
    # Find and count all subtables
    grouped_subtables = find_subtables_with_question(df_data)

    summary_df = None

    # analyze all the subtables and get their values
    for i in range(len(grouped_subtables)):
        try:
            summary_df = analyze_subtable(grouped_subtables, i, summary_df)
        except Exception as e:
            print(f"Error analyzing subtable {i}: {e}")

    print("Summary_df:\n", summary_df)

    # Save the summary_df as a file
    if summary_df is not None:
        output_dir = os.path.join(os.path.dirname(__file__), '../data/custom_files/')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'summary_df.xlsx')
        summary_df.to_excel(output_file)
        print(f"Summary DataFrame saved to {output_file}")

