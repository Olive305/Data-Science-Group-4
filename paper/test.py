import joblib
import pandas as pd
from data_extraction.utils import load_excel
from paper.monte_carlo import monte
from model_outcome import find_optimal_contribution_df, inspect_insurer_df
from paper.monte_execution import simulate_income, simulate_demography


def starting_point():
    df = load_excel('../data/fm_dem_sat_merged.xlsx')
    # Replace missing values with median to avoid NaNs
    df = df.fillna(df.median(numeric_only=True))

    latest_rows = []
    for kk in df['Krankenkasse'].unique():
        df_kk = df[df['Krankenkasse'] == kk]
        latest = df_kk[df_kk['Date'] == df_kk['Date'].max()].copy()
        latest_rows.append(latest)
    df_latest = pd.concat(latest_rows, ignore_index=True)
    # Drop date columns to keep only relevant features
    return df_latest.drop(columns=['Jahr', 'Quartal', 'Date'])


def execute_one_period(df_current, monte_row, model_bundle, insurer):
    """
    Apply Monte Carlo shocks for income and demographics to all insurers,
    optimize the contribution rate for the target insurer,
    and calculate relevant metrics.

    Args:
        df_current: current state dataframe of all insurers
        monte_row: one row from Monte Carlo output with shocks
        model_bundle: loaded predictive model
        insurer: target insurer for optimization

    Returns:
        updated dataframe for next period,
        metrics dict for the target insurer
    """
    rows = []
    insurer_metrics = {}

    for _, kk_row in df_current.iterrows():
        data = kk_row.to_dict()
        # 1) Apply income shock
        econ = monte_row['economy']
        data.update(simulate_income(econ, kk_row))
        # 2) Apply demographic change
        demo_dist, mit_new, ves_new = simulate_demography(monte_row, kk_row)
        data.update(demo_dist)
        data['Mitglieder'] = mit_new
        data['Versicherte'] = ves_new
        # 3) Set base contribution difference from Monte Carlo competitor
        data['ZB_diff'] = monte_row['competitor']
        data['customer'] = monte_row['customer']
        temp_df = pd.DataFrame([data])

        if kk_row['Krankenkasse'] == insurer:
            # Optimize contribution rate specifically for this insurer
            opt = find_optimal_contribution_df(temp_df, model_bundle, insurer)
            data['ZB_diff'] = opt['best_diff']
            metrics = inspect_insurer_df(temp_df, model_bundle, insurer, opt['best_diff'])
            # Calculate peer_mean_net if not directly available
            if 'peer_mean_net' in metrics and metrics['peer_mean_net'] is not None:
                peer_mean_net = metrics['peer_mean_net']
            elif 'diff_vs_others' in metrics and 'net' in metrics:
                peer_mean_net = metrics['net'] - metrics['diff_vs_others']
            else:
                peer_mean_net = None

            insurer_metrics = {
                'optimal_ZB_diff': opt['best_diff'],
                'net': metrics['net'],
                'peer_mean_net': peer_mean_net
            }
        else:
            # For other insurers, predict metrics with current ZB_diff
            metrics = inspect_insurer_df(temp_df, model_bundle, kk_row['Krankenkasse'], kk_row['ZB_diff'])

        data['net'] = metrics['net']
        rows.append(data)

    return pd.DataFrame(rows), insurer_metrics


def run_multi_periods(years=10, insurers=None, model_path='../models/causal_forest_full_honest.pkl'):
    """
    Run simulation for multiple insurers over multiple years.
    Each insurer is optimized independently over the full period.

    Args:
        years: number of years to simulate (4 periods per year)
        insurers: list of insurer names to optimize
        model_path: path to the saved model bundle

    Returns:
        DataFrame with simulation results for all insurers and periods
    """
    if insurers is None:
        insurers = ['aokbadenwürttemberg']  # default insurer

    bundle = joblib.load(model_path)
    periods = years * 4
    all_results = []

    for insurer in insurers:
        df_current = starting_point()
        for i in range(periods):
            df_monte = monte(1, 1)
            monte_row = df_monte.iloc[0]
            df_next, metrics = execute_one_period(df_current, monte_row, bundle, insurer)
            all_results.append({
                'insurer': insurer,
                'period': i + 1,
                'optimal_ZB_diff': metrics.get('optimal_ZB_diff'),
                'net': metrics.get('net'),
                'peer_mean_net': metrics.get('peer_mean_net')
            })
            df_current = df_next.copy()

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    # Example: simulate for two insurers over 10 years
    insurers_list = ['aokbadenwürttemberg', 'tk']  # adjust insurer names as needed
    df_summary = run_multi_periods(years=10, insurers=insurers_list)
    print(df_summary)
