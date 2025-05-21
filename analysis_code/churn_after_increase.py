import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity

#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#linear regression takes the independant and dependant variables
def linear_regression(X,y):
    # train and test
    X_train, X_test, y_train, y_test = (sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=69))

    # scaling the data so that average =0 and standard dev =1
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    model = sk.linear_model.LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = sk.metrics.r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_test.shape[1]

    # Adjusted R²
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    # print results
    print("\n")
    print("Coeficient:", model.coef_)
    print("Intercept ):", model.intercept_)
    print("R²:", r2)
    print("adj R²:", r2_adj)
    print("MSE:", sk.metrics.mean_squared_error(y_test, y_pred))

def data_cleanup(df):
    # combine Year and Quarter for easier calculations
    df['Date'] = pd.PeriodIndex.from_fields(year=df['Jahr'], quarter=df['Quartal'], freq='Q')

    # sort by Name and Date => to calculate the difference in members for next year per insurer
    df = df.sort_values(by=['Krankenkasse', 'Date'])

    # calculate the increase in fees compared to the previous year
    df['Zusatzbeitrag_diff'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].diff()

    # calculate the amount of members lost compared to the year after the current year
    df['Mitglieder_diff_next'] = df.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df['Mitglieder']

    df['Zusatzbeitrag_diff'] = df['Zusatzbeitrag_diff'].fillna(0)
    df['Mitglieder_diff_next'] = df['Mitglieder_diff_next'].fillna(0)
    return(df)

def reg_fee_churn():
    #import data
    df = pd.read_excel('../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
    df = data_cleanup(df)
    linear_regression(df[['Zusatzbeitrag_diff']], df['Mitglieder_diff_next'])

def reg_morb_fee_churn():
    df= fuz_combine_fees_morbidity()
    df = data_cleanup(df)
    linear_regression(df[['Zusatzbeitrag_diff']], df['Mitglieder_diff_next'])


reg_fee_churn()
#print(df)