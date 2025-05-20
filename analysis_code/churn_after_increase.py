import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

#import data
df = pd.read_excel(os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx'))

#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#combine Year and Quarter for easier calculations
df['Date'] = pd.PeriodIndex.from_fields(year=df['Jahr'], quarter=df['Quartal'], freq='Q')

#sort by Name and Date => to calculate the difference in members for next year per insurer
df = df.sort_values(by = ['Krankenkasse','Date'])

#calculate the increase in fees compared to the previous year
df['Zusatzbeitrag_diff'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].diff()

#calculate the amount of members lost compared to the year after the current year
df['Mitglieder_diff_next'] = df.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df['Mitglieder']

df['Zusatzbeitrag_diff'] = df['Zusatzbeitrag_diff'].fillna(0)
df['Mitglieder_diff_next'] = df['Mitglieder_diff_next'].fillna(0)

#change in fee = independant variable; change in membership = dependant variable
X = df[['Zusatzbeitrag_diff']]
y = df['Mitglieder_diff_next']


"""
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

"""
#train and test
X_train, X_test, y_train, y_test = (sk.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 69))

#scaling the data so that average =0 and standard dev =1
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

model = sk.linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2= sk.metrics.r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]

# Adjusted R²
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#print results
print("\n")
print("Coeficient:", model.coef_)
print("Intercept ):", model.intercept_)
print("R²:", r2)
print("adj R²:", r2_adj)
print("MSE:", sk.metrics.mean_squared_error(y_test, y_pred))
