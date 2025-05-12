import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import os


#import data
location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df = pd.read_excel(location)


#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#combine Year and Quarter for easier calculations
df['Date'] = pd.to_datetime(df['Jahr'].astype(str) + 'Q'+ df['Quartal'].astype(str))

#sort by Name and Date => to calculate the difference in members for next year per insurer
df = df.sort_values(by = ['Krankenkasse','Date'])

# Ensure 'Zusatzbeitrag' is numeric and fill NaN values with 0
df['Zusatzbeitrag'] = pd.to_numeric(df['Zusatzbeitrag'], errors='coerce').fillna(0)

# Calculate the increase/decrease in fees as a percentage compared to the previous year
df['Zusatzbeitrag_diff'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].pct_change()
df['Zusatzbeitrag_diff'] = df['Zusatzbeitrag_diff'].fillna(0)

#calculate the percentage difference in members compared to the year after the current year
df['Mitglieder_diff_next'] = (df.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df['Mitglieder']) / df['Mitglieder']

#change in fee = independant variable; change in membership = dependant variable
X = df[['Zusatzbeitrag_diff']]
y = df['Mitglieder_diff_next']

# Show the values (x and y) in a graph to look at their coherence
# Each value pair as a dot in the graph
if False: 
    plt.scatter(X, y, alpha=0.5)
    plt.title('Zusatzbeitrag_diff vs Mitglieder_diff_next')
    plt.xlabel('Zusatzbeitrag_diff')
    plt.ylabel('Mitglieder_diff_next')
    plt.grid(True)
    plt.show()

# Print the first 5 rows of the dataframe
print(df.head())

# Create a table which shows the price increase over time
if True:
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('Krankenkasse'):
        plt.plot(group['Date'], group['Zusatzbeitrag_diff'], marker='o', label=name, alpha=0.7)

    plt.title('Price Increase Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price Increase (Zusatzbeitrag_diff)')
    plt.legend(loc='best', fontsize='small', title='Krankenkasse')
    plt.grid(True)
    plt.show()


#train and test
X_train, X_test, y_train, y_test = (sk.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 69))


model = sk.linear_model.LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
#print results
print("\n")
print("Coeficient:", model.coef_)
print("Intercept ):", model.intercept_)
print("R²:", sk.metrics.r2_score(y_test, y_pred))
print("MSE:", sk.metrics.mean_squared_error(y_test, y_pred))

print(df)