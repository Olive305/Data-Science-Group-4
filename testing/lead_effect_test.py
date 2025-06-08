import numpy as np
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV


def lead_effect_test(df, feature_cols, treatment_col, outcome_col, time_col, lead=1):
    df_lead = df.copy()
    df_lead[treatment_col + "_lead"] = df.groupby("Krankenkasse")[treatment_col].shift(-lead)
    df_lead = df_lead.dropna()


    X = df_lead[feature_cols]
    T = df_lead[treatment_col + "_lead"]
    Y = df_lead[outcome_col]

    cf = CausalForestDML(model_y=LassoCV(), model_t=LassoCV(), discrete_treatment=False, random_state=0)
    cf.fit(Y, T, X=X)
    lead_effect = np.mean(cf.effect(X))
    return lead_effect
