import pandas as pd
import numpy as np

from data_extraction.utils import normalize_features

# model_bundle: dict with keys 'model', 'scaler', 'features'

def compute_avg_persons(df_q: pd.DataFrame, insurer: str) -> float:
    """
    Calculate average household size for insurer.
    """
    row = df_q[df_q['Krankenkasse'] == insurer]
    if row.empty:
        raise ValueError(f"Keine Daten für {insurer!r}")
    p1, p2, p3_4 = row.iloc[0][['Personen im Haushalt_1', 'Personen im Haushalt_2', 'Personen im Haushalt_3-4']]
    rest = 1.0 - (p1 + p2 + p3_4)
    return p1*1 + p2*2 + p3_4*3.5 + rest*5


def apply_contribution(df_q: pd.DataFrame, insurer: str, delta: float, col: str = 'ZB_diff'):
    """
    Set contribution change for insurer in dataframe.
    """
    if col not in df_q.columns:
        df_q[col] = 0.0
    df_q[col] = df_q[col].fillna(0.0)
    mask = df_q['Krankenkasse'] == insurer
    if not mask.any():
        raise ValueError(f"{insurer!r} not found")
    df_q.loc[mask, col] = delta


def distribute_costs_linear() -> dict:
    """
    Load age-group costs and convert to monthly per-capita.
    """
    df = pd.read_excel('../data/Krankkosten.xlsx')
    age_groups = [
        '15 bis unter 30 Jahre', '30 bis unter 45 Jahre',
        '45 bis unter 65 Jahre', '65 bis unter 85 Jahre', '85 Jahre und älter'
    ]
    costs = df[age_groups].iloc[0]
    return {
        'Alter_16-29 Jahre': costs['15 bis unter 30 Jahre']/12,
        'Alter_30-39 Jahre': (costs['30 bis unter 45 Jahre']/15*10)/12,
        'Alter_40-49 Jahre': ((costs['30 bis unter 45 Jahre']/15*5 + costs['45 bis unter 65 Jahre']/20*5)/12),
        'Alter_50-59 Jahre': (costs['45 bis unter 65 Jahre']/20*10)/12,
        'Alter_60-69 Jahre': ((costs['45 bis unter 65 Jahre']/20*5 + costs['65 bis unter 85 Jahre']/20*5)/12),
        'Alter ≥ 70 Jahre': ((costs['65 bis unter 85 Jahre']/20*15 + costs['85 Jahre und älter'])/12)
    }


def calc_financials(
    row: pd.Series,
    diff: float,
    delta: float,
    basis_zb: float,
    avg_persons: float
) -> dict:
    """
    Compute revenue, cost, net, margin for one insurer.
    """
    # income midpoints
    inc_mid = {
        'Unter 1.000': 500, '1.000-1.499': 1250,
        '1.500-1.999': 1750, '2.000-2.499': 2250,
        '2.500-3.999': 3250, '>3.999': 4500
    }
    inc_cols = {
        'Haushaltsnettoeinkommen in Euro_Unter 1.000': 'Unter 1.000',
        'Haushaltsnettoeinkommen in Euro_1.000-1.499': '1.000-1.499',
        'Haushaltsnettoeinkommen in Euro_1.500-1.999': '1.500-1.999',
        'Haushaltsnettoeinkommen in Euro_2.000-2.499': '2.000-2.499',
        'Haushaltsnettoeinkommen in Euro_2.500-3.999': '2.500-3.999'
    }
    shares = row[list(inc_cols.keys())]
    rest = 1 - shares.sum()
    factor = avg_persons ** 0.5
    avg_income = sum(shares[c] * inc_mid[l] / factor for c, l in inc_cols.items()) + rest * inc_mid['>3.999'] / factor
    avg_income *= 1.4

    mit_new = row['Mitglieder'] + diff
    revenue = delta * mit_new * avg_income

    age_shares = {k: row[k] for k in [
        'Alter_16-29 Jahre','Alter_30-39 Jahre','Alter_40-49 Jahre',
        'Alter_50-59 Jahre','Alter_60-69 Jahre'
    ]}
    age_shares['Alter ≥ 70 Jahre'] = 1 - sum(age_shares.values())
    cost_map = distribute_costs_linear()
    total_cost = sum(age_shares[seg] * cost_map[seg] * mit_new for seg in age_shares)

    loss = total_cost + (basis_zb + delta) * abs(diff) * avg_income
    net = revenue - loss
    return {
        'avg_income': avg_income,
        'revenue': revenue,
        'cost': total_cost,
        'loss': loss,
        'net': net,
        'profit_margin': net / revenue if revenue != 0 else 0
    }


def optimize_contribution_in_df(
    df_q: pd.DataFrame,
    model_bundle: dict,
    insurer: str,
    zb_range: tuple = (-0.5, 1.0),
    step: float = 0.1,
    col: str = 'ZB_diff'
) -> dict:
    """
    Find optimal ZB_diff for insurer, update df_q, return metrics.

    Returns dict with keys:
      - best_diff, net, profit_margin, compare_to_others, updated_df
    """
    # ensure column
    if col not in df_q.columns:
        df_q[col] = 0.0
    df_q[col] = df_q[col].fillna(0.0)

    # backup base contributions
    basis_zb_map = df_q.set_index('Krankenkasse')['Zusatzbeitrag'].to_dict()

    # prepare prediction
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    features = model_bundle['features']
    df_mod = df_q.copy()
    df_mod['predicted_diff'] = model.predict(
        pd.DataFrame(scaler.transform(df_mod[features]), columns=features, index=df_mod.index)
    ).ravel()

    # compute peers base net
    net_base = {}
    for _, row in df_mod.iterrows():
        name = row['Krankenkasse']
        avg_p = compute_avg_persons(df_q, name)
        fin = calc_financials(row, row['predicted_diff'], 0.0, basis_zb_map[name], avg_p)
        net_base[name] = fin['net']
    peer_mean_base = np.mean([v for k,v in net_base.items() if k != insurer])

    best = None
    best_diff = None

    for delta in np.arange(zb_range[0], zb_range[1] + step, step):
        row = df_mod[df_mod['Krankenkasse']==insurer].iloc[0]
        avg_p = compute_avg_persons(df_q, insurer)
        fin = calc_financials(row, row['predicted_diff'], delta, basis_zb_map[insurer], avg_p)
        if best is None or fin['net'] > best['net']:
            best = fin
            best_diff = round(delta,2)

    # apply optimal and update Mitglieder
    apply_contribution(df_q, insurer, best_diff, col)
    df_q.loc[df_q['Krankenkasse']==insurer, 'Mitglieder'] += df_mod.loc[df_mod['Krankenkasse']==insurer, 'predicted_diff'].iloc[0]

    compare_to_others = best['net'] - peer_mean_base

    return {
        'best_diff': best_diff,
        'net': best['net'],
        'profit_margin': best['profit_margin'],
        'compare_to_others': compare_to_others,
        'updated_df': df_q
    }
