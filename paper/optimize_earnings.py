import numpy as np
import pandas as pd

def predict_membership_change_full(df, model_bundle, contrib_change):
    """
    Predict relative membership change per row in df after applying contribution change using trained CF model.

    Parameters:
    - df (pd.DataFrame): Full DataFrame with all features used in model training
    - model_bundle (dict): Dict containing trained model, scaler and feature list
    - contrib_change (float): Contribution change to simulate (e.g. 0.005 for +0.5%)

    Returns:
    - pd.DataFrame: Input df augmented with columns:
        - 'predicted_cate': estimated individual treatment effect (relative change)
        - 'predicted_relative_change': same as cate for clarity
        - 'predicted_members': estimated new membership number after change
    """

    # Extract features & scaler
    features = model_bundle["features"]
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]

    # Prepare feature matrix, normalize as model expects
    X = df[features]
    X_norm = scaler.transform(X)

    # Calculate individual treatment effects (CATEs)
    cates = model.effect(X_norm)

    # Attach CATE predictions to DataFrame
    df = df.copy()
    df["predicted_cate"] = cates
    df["predicted_relative_change"] = cates

    # Calculate predicted membership after contribution change
    # Assumes df contains a 'Mitglieder' column with current member counts
    df["predicted_members"] = df["Mitglieder"] * (1 + df["predicted_relative_change"])

    return df


def cost_calculation(df, cost_per_age):
    """
    Calculate estimated total cost based on predicted member counts and cost per age group.

    Parameters:
    - df (pd.DataFrame): DataFrame including predicted_members and 'Alter' columns
    - cost_per_age (dict): Mapping from age to cost per member in that age group

    Returns:
    - float: Estimated total cost
    """
    total_cost = 0
    for age, cost in cost_per_age.items():
        # Sum predicted members by age * cost
        members_age = df.loc[df["Alter"] == age, "predicted_members"].sum()
        total_cost += members_age * cost

    return total_cost


def revenue_calculation(df, base_contrib, contrib_increase):
    """
    Calculate revenue based on predicted members and contribution rate.

    Parameters:
    - df (pd.DataFrame): DataFrame including predicted_members
    - base_contrib (float): baseline contribution rate (e.g. 14.5)
    - contrib_increase (float): increase in contribution rate (e.g. 0.005)

    Returns:
    - float: Estimated total revenue
    """
    contrib = base_contrib + contrib_increase
    total_members = df["predicted_members"].sum()
    revenue = contrib * total_members
    return revenue


def max_earnings(df, model_bundle, cost_per_age, base_contrib=14.5, max_increase_pct=0.5, steps=50):
    """
    Calculate max profit scenario by varying contribution increase from 0 to max_increase_pct.

    Parameters:
    - df (pd.DataFrame): full input DataFrame with features and current membership
    - model_bundle (dict): trained causal forest model bundle
    - cost_per_age (dict): cost estimates per age group
    - base_contrib (float): base contribution rate
    - max_increase_pct (float): maximum contribution increase to consider (e.g. 0.5 = 50%)
    - steps (int): number of increments in range

    Returns:
    - list of tuples: [(contrib_increase, profit), ...]
    """
    results = []

    for step in range(steps + 1):
        contrib_increase = max_increase_pct * (step / steps)
        # Predict membership change for given contrib increase
        pred_df = predict_membership_change_full(df, model_bundle, contrib_increase)

        # Calculate revenue and costs
        revenue = revenue_calculation(pred_df, base_contrib, contrib_increase)
        cost = cost_calculation(pred_df, cost_per_age)

        profit = revenue - cost
        results.append((contrib_increase, profit))

    return results
