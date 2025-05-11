import pandas as pd
import sklearn as sk
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

#calculate the increase in fees compared to the previous year
df['Zusatzbeitrag_diff'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].diff()

#calculate the amount of members lost compared to the year after the current year
df['Mitglieder_diff_next'] = df.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df['Mitglieder']

df['Zusatzbeitrag_diff'] = df['Zusatzbeitrag_diff'].fillna(0)
df['Mitglieder_diff_next'] = df['Mitglieder_diff_next'].fillna(0)

#change in fee = independant variable; change in membership = dependant variable
X = df[['Zusatzbeitrag_diff']]
y = df['Mitglieder_diff_next']

#train and test
X_train, X_test, y_train, y_test = (sk.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 69))


model = sk.linear_model.LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
#print results
print("Coeficient:", model.coef_)
print("Intercept ):", model.intercept_)
print("R²:", sk.metrics.r2_score(y_test, y_pred))
print("MSE:", sk.metrics.mean_squared_error(y_test, y_pred))



print(df)