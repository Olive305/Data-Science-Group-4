import os
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# import data
location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
df_Kundenmonitor = pd.read_excel(location, sheet_name="EE")
location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_churn = pd.read_excel(location)

print(df_Kundenmonitor.head)

#combine Year and Quarter for easier calculations
df_churn['Date'] = pd.to_datetime(df_churn['Jahr'].astype(str) + 'Q'+ df_churn['Quartal'].astype(str))

#calculate the percentage difference in members compared to the year after the current year
df_churn['Mitglieder_diff_next'] = (df_churn.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df_churn['Mitglieder']) / df_churn['Mitglieder']

# Calculate the average churn rate for each insurance company
average_churn = df_churn.groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()

# Create a new DataFrame with the average churn values
df_average_churn = average_churn.rename(columns={'Mitglieder_diff_next': 'Average_Churn_Rate'})

# Add a column to the dataframe with the churn rate in 2023
df_churn_2023 = df_churn[df_churn['Jahr'] == 2023].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2023 = df_churn_2023.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2023'})
df_average_churn = pd.merge(df_average_churn, df_churn_2023, on='Krankenkasse', how='left')

# Now we add the values of the df_Kundenmonitor into the df_average_churn dataframe such that same insurance companys are together
# Merge the satisfaction data from df_Kundenmonitor into df_average_churn
# Transpose df_Kundenmonitor for easier merging
df_Kundenmonitor_transposed = df_Kundenmonitor.set_index('Unnamed: 0').transpose()

# Reset index and rename columns for merging
df_Kundenmonitor_transposed = df_Kundenmonitor_transposed.reset_index().rename(columns={'index': 'Krankenkasse'})

# Merge the transposed Kundenmonitor data into df_average_churn
df_average_churn = pd.merge(df_average_churn, df_Kundenmonitor_transposed, on='Krankenkasse', how='left')

# Now we look for a linear correlation between the values of the Kundenmonitor file and the churn rate
# We create a linear regression model for this and check r squared

# Iterate through each satisfaction column and compare it with the churn rate
satisfaction_columns = df_Kundenmonitor_transposed.columns[3:]

for column in satisfaction_columns:
    print(f"Analyzing satisfaction column: {column}")
    
    # Extract the current satisfaction column and churn rates
    X = df_average_churn[[column]].dropna()  # Drop rows with NaN values
    y = df_average_churn.loc[X.index, 'Churn_Rate_2023']  # Align y with non-NaN rows in X

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R-squared for {column}: {r2}")
    print(f"Mean Squared Error for {column}: {mse}")
    print("-" * 50)



print(df_average_churn)