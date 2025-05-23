import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
def linear_regression(X, y, name, seeds=range(100)):

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
    print(f"Intercept: {np.mean(intercepts):.4f} ± {np.std(intercepts):.4f}")
    print("Coefficients (mean ± std):")
    for i, col in enumerate(X.columns):
        mean_coef = np.mean(coefs[:, i])
        std_coef = np.std(coefs[:, i])
        print(f"  {col}: {mean_coef:.4f} ± {std_coef:.4f}")

def data_cleanup(df):
    # combine Year and Quarter for easier calculations
    df['Date'] = pd.PeriodIndex.from_fields(year=df['Jahr'], quarter=df['Quartal'], freq='Q')

    # sort by Name and Date => to calculate the difference in members for next year per insurer
    df = df.sort_values(by=['Krankenkasse', 'Date'])

    # calculate the increase in fees compared to the previous year
    df['Zusatzbeitrag_diff'] = df.groupby('Krankenkasse')['Zusatzbeitrag'].diff()

    # calculate the amount of members lost compared to the year after the current year
    df['Mitglieder_diff_next'] = df.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df['Mitglieder']
    #df['Mitglieder_diff_next'] = df.groupby('Krankenkasse')['Mitglieder'].diff()

    """dropping the empty ones here for the following reason:
    we try to use the change in this quarters contribution to predict
    the change in Members for the next quarter as we expect people to switch AFTER
    the change not before it and thus this would only present in the quarter after the change occured
    """
    df = df.dropna(subset=['Zusatzbeitrag_diff'])
    df = df.dropna(subset=['Mitglieder_diff_next'])
    #df['Zusatzbeitrag_diff'] = df['Zusatzbeitrag_diff'].fillna(0)
    #df['Mitglieder_diff_next'] = df['Mitglieder_diff_next'].fillna(0)

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

    #cleanup for when there is no data and thus -
    df['Risikofaktor'] = (
        df['Risikofaktor'].astype(str)
        .str.replace('-', '1', regex=False)
        .str.replace('–', '1', regex=False)
    )
    #convert back to float as the conversion was just for cleanup
    df['Risikofaktor'] = pd.to_numeric(df['Risikofaktor'], errors='coerce')
    #certain data points are 0 which makes no sense thus they are dropped
    df = df[df['Risikofaktor'] != 0]
    #drop the ones where there was no data for Risikofaktor
    df = df.dropna(subset=['Risikofaktor'])
    df['MGxRF']    = ((df['Mitglieder'] * df['Risikofaktor'])/4) #interactive term
    #df['Family_Quote'] = df['Versicherte']/df['Mitglieder']
    #linear regression
    linear_regression(df[['Zusatzbeitrag_diff', 'Risikofaktor','Mitglieder','MGxRF','Versicherte']], df['Mitglieder_diff_next'], "morb_fee_churn:")
    return(df)

def clustering():
    df= reg_morb_fee_churn()
    cluster_feats = df[['Mitglieder', 'Risikofaktor', 'Zusatzbeitrag_diff']]

    scaler = StandardScaler()
    Xc = scaler.fit_transform(cluster_feats)


    kmeans = KMeans(n_clusters=3, random_state=42).fit(Xc)
    df['cluster'] = kmeans.labels_

    for g, sub in df.groupby('cluster'):
        print(f"\nCluster {g}: n={len(sub)}")
        linear_regression(sub[['Zusatzbeitrag_diff', 'Mitglieder', 'Risikofaktor', 'MGxRF']],
                          sub['Mitglieder_diff_next'], name=f"Cluster {g}")



def random_forest_regression(X, y, name="Random Forest", do_print=True, seeds=range(10)):
    r2_list = []
    mse_list = []
    feature_importance_list = []
    preds_examples = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = sk.ensemble.RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2_list.append(sk.metrics.r2_score(y_test, y_pred))
        mse_list.append(sk.metrics.mean_squared_error(y_test, y_pred))
        feature_importance_list.append(model.feature_importances_)


        sample_indices = np.random.choice(len(y_test), size=min(3, len(y_test)), replace=False)
        for idx in sample_indices:
            preds_examples.append({
                'y_true': y_test.iloc[idx],
                'y_pred': y_pred[idx],
                'features': dict(zip(X.columns, X_test.iloc[idx]))
            })

    mean_importances = np.mean(feature_importance_list, axis=0)
    importances = dict(zip(X.columns, mean_importances))

    if do_print:
        print(f"\n{name}")
        print(f"R²:  {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
        print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")

        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.3f}")

        for ex in preds_examples:
            print(f"  y_true = {ex['y_true']:.1f}, y_pred = {ex['y_pred']:.1f}, features = {ex['features']}")
    else:
        return {
            'R2': np.mean(r2_list),
            'MSE': np.mean(mse_list),
            'Importances': importances,
            'Example_Predictions': preds_examples
        }


def morb_forest():
    df = reg_morb_fee_churn()
    X = df[['Zusatzbeitrag_diff', 'Risikofaktor', 'Mitglieder', 'MGxRF', 'Versicherte']]
    y = df['Mitglieder_diff_next']

    random_forest_regression(X, y, name="RF: Morbiditäts-Modell")

reg_fee_churn()
#morb_forest()
reg_morb_fee_churn()