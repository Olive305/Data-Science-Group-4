import joblib
import pandas as pd
from data_extraction.utils import load_excel
from paper.monte_carlo import monte
from paper.model_outcome import optimize_contribution_in_df
from paper.monte_execution import simulate_income, simulate_demography


def starting_point(is_did=False):
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    df = df.fillna(df.median(numeric_only=True))
    latest_rows = []
    for kk in df['Krankenkasse'].unique():
        df_kk = df[df['Krankenkasse'] == kk]
        latest = df_kk[df_kk['Date'] == df_kk['Date'].max()].copy()
        latest_rows.append(latest)
    df_latest = pd.concat(latest_rows, ignore_index=True)
    if is_did:
        return df_latest.drop(columns=['Jahr', 'Quartal'])
    return df_latest.drop(columns=['Jahr', 'Quartal', 'Date'])


def execute_one_period(df_current, monte_row, model_bundle, insurer):
    """
    Apply Monte Carlo shocks to all insurers, then optimize the target insurer.
    Returns the next period DataFrame and metrics for the target insurer.
    """
    # 1) Create shocked DataFrame
    shocked = []
    for _, row in df_current.iterrows():
        data = row.to_dict()
        # income shock
        data.update(simulate_income(monte_row['economy'], row))
        # demographic change
        demo_dist, mit_new, ves_new = simulate_demography(monte_row, row)
        data.update(demo_dist)
        data['Mitglieder'] = mit_new
        data['Versicherte'] = ves_new
        shocked.append(data)
    df_shock = pd.DataFrame(shocked)

    # 2) Set initial competitor diff for all
    df_shock['ZB_diff'] = monte_row['competitor']

    # 3) Optimize contribution in DataFrame
    result = optimize_contribution_in_df(df_shock, model_bundle, insurer)
    df_next = result['updated_df']

    # 4) Extract metrics
    best_diff = result['best_diff']
    net = result['net']
    compare_to_others = result['compare_to_others']
    peer_mean = net - compare_to_others

    metrics = {
        'optimal_ZB_diff': best_diff,
        'net': net,
        'peer_mean_net': peer_mean
    }
    return df_next, metrics


def run_multi_periods(years=10, insurers=None, model_path='../models/causal_forest_full_honest.pkl'):
    """
    Run Monte Carlo simulation over multiple periods for given insurers.
    """
    if insurers is None:
        insurers = ['aokbadenwürttemberg']

    bundle = joblib.load(model_path)
    periods = years * 4
    all_results = []

    for insurer in insurers:
        df_current = starting_point()
        for period in range(1, periods+1):
            df_monte = monte(1, 1).iloc[0]
            df_next, metrics = execute_one_period(df_current, df_monte, bundle, insurer)
            all_results.append({
                'insurer': insurer,
                'period': period,
                **metrics
            })
            df_current = df_next.copy()

    return pd.DataFrame(all_results)


def full_monte():
    df = starting_point()
    insurers_list = df['Krankenkasse'].unique()
    df_summary = run_multi_periods(years=1, insurers=insurers_list)
    print(df_summary)

if __name__ == '__main__':

    full_monte()