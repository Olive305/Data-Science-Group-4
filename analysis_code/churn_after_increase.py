import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
#show all of the data with print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#linear regression takes the independant and dependant variables
def linear_regression(X, y, name, seeds=range(50)):


    r2_list = []
    r2_adj_list = []
    mse_list = []
    coef_list = []
    intercept_list = []

    print(f"\n{name}")
    print(f"Testing {len(seeds)} different random states...\n")

    for seed in seeds:
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        model = sk.linear_model.LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = sk.metrics.r2_score(y_test, y_pred)
        n = X_test.shape[0]
        p = X_test.shape[1]
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)

        r2_list.append(r2)
        r2_adj_list.append(r2_adj)
        mse_list.append(mse)
        coef_list.append(model.coef_)
        intercept_list.append(model.intercept_)

    coefs = np.vstack(coef_list)
    intercepts = np.array(intercept_list)

    print(f"R²:        {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"Adj R²:    {np.mean(r2_adj_list):.4f} ± {np.std(r2_adj_list):.4f}")
    print(f"MSE:       {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
    print(f"Coef:      {np.mean(coefs, axis=0)} ± {np.std(coefs, axis=0)}")
    print(f"Intercept: {np.mean(intercepts):.4f} ± {np.std(intercepts):.4f}")

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

    #print(df['Krankenkasse'].unique())
    return(df)

def reg_fee_churn():
    #import data
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), '..', 'data', 'Zusatzbeitrag_je Kasse je Quartal.xlsx'))
    df = data_cleanup(df)
    linear_regression(df[['Zusatzbeitrag_diff']], df['Mitglieder_diff_next'],"fee churn:")

def reg_morb_fee_churn():
    df= fuz_combine_fees_morbidity()
    df = df.dropna(subset=['Zusatzbeitrag'])
    df = data_cleanup(df)
    #return df
    linear_regression(df[['Zusatzbeitrag_diff']], df['Mitglieder_diff_next'], "morb_fee_churn:")


reg_fee_churn()
reg_morb_fee_churn()
