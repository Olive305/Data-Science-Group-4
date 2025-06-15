import pandas as pd
import joblib
from data_extraction.utils import load_excel

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def compute_avg_persons(df_q: pd.DataFrame, insurer: str) -> float:
    """
    Calculate the average household size for a given insurer.

    Args:
        df_q: DataFrame containing current period data.
        insurer: The insurer identifier (Krankenkasse).

    Returns:
        The average number of persons per household.
    """
    row = df_q[df_q["Krankenkasse"] == insurer]
    if row.empty:
        raise ValueError(f"No data available for insurer {insurer!r}")

    # Extract shares of households of size 1, 2, and 3-4
    p1, p2, p3_4 = row.iloc[0][[
        "Personen im Haushalt_1",
        "Personen im Haushalt_2",
        "Personen im Haushalt_3-4"
    ]]

    # Remaining share represents households size >=5
    rest = 1.0 - (p1 + p2 + p3_4)

    # Compute weighted average: 1*p1 + 2*p2 + 3.5*p3_4 + 5*rest
    return p1*1 + p2*2 + p3_4*3.5 + rest*5


def apply_contribution(df_q: pd.DataFrame, insurer: str, delta: float, col: str = 'ZB_diff'):
    """
    Apply a manual contribution change for a specific insurer.

    Args:
        df_q: DataFrame containing current period data.
        insurer: The insurer identifier.
        delta: The contribution change to apply.
        col: The column name to store the delta.
    """
    mask = df_q["Krankenkasse"] == insurer
    if not mask.any():
        raise ValueError(f"Insurer {insurer!r} not found")
    df_q.loc[mask, col] = delta


def predict_diff(df_q: pd.DataFrame, model, scaler, features: list) -> pd.DataFrame:
    """
    Add model-based predicted change in membership for each insurer.

    Args:
        df_q: DataFrame containing features.
        model: Trained predictive model.
        scaler: Feature scaler used during training.
        features: List of feature column names.

    Returns:
        A new DataFrame with a 'predicted_diff' column.
    """
    Xn = scaler.transform(df_q[features])
    df_q = df_q.copy()
    df_q['predicted_diff'] = model.predict(
        pd.DataFrame(Xn, columns=features, index=df_q.index)
    ).ravel()
    return df_q


def calc_segment_changes(row: pd.Series, diff: float) -> dict:
    """
    Compute membership changes per age segment based on total difference.

    Args:
        row: A row of DataFrame with age segment shares.
        diff: Total predicted change in membership.

    Returns:
        Dictionary mapping each age segment to its integer change.
    """
    age_keys = [
        "Alter_16-29 Jahre",
        "Alter_30-39 Jahre",
        "Alter_40-49 Jahre",
        "Alter_50-59 Jahre",
        "Alter_60-69 Jahre"
    ]
    segments = {k: row[k] for k in age_keys}
    rest_share = 1 - sum(segments.values())
    segments['Alter ≥ 70 Jahre'] = rest_share

    # Allocate integer changes proportionally
    return {seg: int(round(share * diff)) for seg, share in segments.items()}


def distribute_costs_linear() -> dict:
    """
    Load cost per age group and distribute linearly over months.

    Returns:
        Mapping from age segment to monthly per-capita cost.
    """
    df = load_excel('../data/Krankkosten.xlsx')
    age_groups = [
        "15 bis unter 30 Jahre",
        "30 bis unter 45 Jahre",
        "45 bis unter 65 Jahre",
        "65 bis unter 85 Jahre",
        "85 Jahre und älter"
    ]
    costs = df[age_groups].iloc[0]

    # Convert annual costs to monthly per-capita by segment
    return {
        "Alter_16-29 Jahre": costs["15 bis unter 30 Jahre"]/12,
        "Alter_30-39 Jahre": (costs["30 bis unter 45 Jahre"]/15*10)/12,
        "Alter_40-49 Jahre": ((costs["30 bis unter 45 Jahre"]/15*5 + costs["45 bis unter 65 Jahre"]/20*5)/12),
        "Alter_50-59 Jahre": (costs["45 bis unter 65 Jahre"]/20*10)/12,
        "Alter_60-69 Jahre": ((costs["45 bis unter 65 Jahre"]/20*5 + costs["65 bis unter 85 Jahre"]/20*5)/12),
        "Alter ≥ 70 Jahre": ((costs["65 bis unter 85 Jahre"]/20*15 + costs["85 Jahre und älter"])/12)
    }


def calc_financials(
    row: pd.Series,
    diff: float,
    delta: float,
    basis_zb: float,
    avg_persons: float
) -> dict:
    """
    Calculate financial metrics (revenue, cost, net, margin) for an insurer.

    Args:
        row: Data row containing demographic and income shares.
        diff: Predicted membership change.
        delta: Contribution change applied.
        basis_zb: Base contribution before change.
        avg_persons: Average household size.

    Returns:
        Dictionary with avg_income, revenue, cost, loss, net, profit_margin.
    """
    # Midpoints for income brackets
    inc_mid = {
        "Unter 1.000": 500, "1.000-1.499": 1250,
        "1.500-1.999": 1750, "2.000-2.499": 2250,
        "2.500-3.999": 3250, ">3.999": 4500
    }
    inc_cols = {
        "Haushaltsnettoeinkommen in Euro_Unter 1.000": "Unter 1.000",
        "Haushaltsnettoeinkommen in Euro_1.000-1.499": "1.000-1.499",
        "Haushaltsnettoeinkommen in Euro_1.500-1.999": "1.500-1.999",
        "Haushaltsnettoeinkommen in Euro_2.000-2.499": "2.000-2.499",
        "Haushaltsnettoeinkommen in Euro_2.500-3.999": "2.500-3.999"
    }
    shares = row[list(inc_cols.keys())]
    rest = 1 - shares.sum()
    factor = avg_persons ** 0.5

    # Compute average income per person
    avg_income = sum(
        shares[c] * inc_mid[l] / factor for c, l in inc_cols.items()
    ) + rest * inc_mid[">3.999"] / factor
    avg_income *= 1.4  # apply additional scaling

    # Membership count
    mit_old = row['Mitglieder']
    mit_new = mit_old + diff

    # Revenue = contribution * new members * avg income
    revenue = delta * mit_new * avg_income

    # Compute age-weighted cost
    age_keys = [
        "Alter_16-29 Jahre", "Alter_30-39 Jahre",
        "Alter_40-49 Jahre", "Alter_50-59 Jahre", "Alter_60-69 Jahre"
    ]
    age_shares = {k: row[k] for k in age_keys}
    age_shares["Alter ≥ 70 Jahre"] = 1 - sum(age_shares.values())

    cost_map = distribute_costs_linear()
    total_cost = sum(
        age_shares[seg] * cost_map[seg] * mit_new for seg in age_shares
    )

    # Loss = cost + absolute membership change * avg contribution * avg_income
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


def inspect_insurer_df(
    df_q: pd.DataFrame,
    model_bundle: dict,
    insurer: str,
    delta: float
) -> dict:
    """
    Evaluate financial performance for a single insurer in-place.

    Args:
        df_q: DataFrame with current period data.
        model_bundle: Dict containing 'model', 'scaler', 'features'.
        insurer: Insurer identifier to inspect.
        delta: Contribution change for this insurer.

    Returns:
        Dict of financial results, segment changes, and comparison to peers.
    """
    avg_p = compute_avg_persons(df_q, insurer)
    base_row = df_q[df_q['Krankenkasse'] == insurer].iloc[0]
    basis_zb = base_row['Zusatzbeitrag']

    # Apply manual contribution change
    apply_contribution(df_q, insurer, delta)  # writes 'ZB_diff'

    # Predict membership change
    df_pred = predict_diff(
        df_q, model_bundle['model'], model_bundle['scaler'], model_bundle['features']
    )

    # Extract metrics for target insurer
    row = df_pred[df_pred['Krankenkasse'] == insurer].iloc[0]
    diff = row['predicted_diff']

    seg_changes = calc_segment_changes(row, diff)
    fin = calc_financials(row, diff, delta, basis_zb, avg_p)

    # Compare to peer average
    peer_nets = []
    for _, r in df_pred[df_pred['Krankenkasse'] != insurer].iterrows():
        avg_p_o = compute_avg_persons(df_q, r['Krankenkasse'])
        fin_o = calc_financials(r, r['predicted_diff'], 0.0, r['Zusatzbeitrag'], avg_p_o)
        peer_nets.append(fin_o['net'])

    if peer_nets:
        peers_mean = sum(peer_nets) / len(peer_nets)
    else:
        peers_mean = 0

    diff_vs_peers = fin['net'] - peers_mean

    result = {**fin, 'segments': seg_changes, 'diff_vs_others': diff_vs_peers}
    return result


def find_optimal_contribution_df(
    df_q: pd.DataFrame,
    model_bundle: dict,
    insurer: str,
    zb_range=(-0.5, 1.0),
    step=0.1
) -> dict:
    """
    Determine the optimal contribution change for a target insurer based on
    profit (net) maximization.

    Args:
        df_q: DataFrame with current period data.
        model_bundle: Dict containing 'model', 'scaler', 'features'.
        insurer: Insurer identifier to optimize.
        zb_range: Tuple (min, max) search range for contribution delta.
        step: Increment for grid search.

    Returns:
        Dict with best_diff, net, profit_margin, and comparison to peers.
    """
    # 1) Baseline prediction
    df_pred = predict_diff(
        df_q.copy(),
        model_bundle['model'],
        model_bundle['scaler'],
        model_bundle['features']
    )
    basis_map = df_q.set_index('Krankenkasse')['Zusatzbeitrag'].to_dict()

    # 2) Compute baseline net profits per KK
    base_nets = {}
    for _, r in df_pred.iterrows():
        avg_p = compute_avg_persons(df_q, r['Krankenkasse'])
        fin = calc_financials(
            r,
            r['predicted_diff'],
            0.0,
            basis_map[r['Krankenkasse']],
            avg_p
        )
        base_nets[r['Krankenkasse']] = fin['net']

    # 3) Peer‐average guard (avoid division by zero)
    num_peers = len(base_nets) - 1
    if num_peers > 0:
        peers_mean_base = sum(
            v for k, v in base_nets.items() if k != insurer
        ) / num_peers
    else:
        peers_mean_base = 0

    # 4) Grid‐search for optimal delta
    best = None
    best_diff = None
    for zb_diff in [round(i * step + zb_range[0], 2)
                    for i in range(int((zb_range[1] - zb_range[0]) / step) + 1)]:
        # apply and re‐predict for this delta
        temp = df_pred.copy()
        temp.loc[temp['Krankenkasse'] == insurer, 'ZB_diff'] = zb_diff
        temp = predict_diff(
            temp,
            model_bundle['model'],
            model_bundle['scaler'],
            model_bundle['features']
        )
        row = temp[temp['Krankenkasse'] == insurer].iloc[0]
        avg_p = compute_avg_persons(df_q, insurer)
        fin = calc_financials(
            row,
            row['predicted_diff'],
            zb_diff,
            basis_map[insurer],
            avg_p
        )
        if best is None or fin['net'] > best['net']:
            best = fin.copy()
            best_diff = zb_diff

    return {
        'best_diff': best_diff,
        'net': best['net'],
        'profit_margin': best['profit_margin'],
        'compare_to_others': best['net'] - peers_mean_base
    }
