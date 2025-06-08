from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import numpy as np

def placebo_test(X, T, Y, model_features, n_iter=100):
    placebo_effects = []
    for _ in range(n_iter):
        T_placebo = np.random.permutation(T)  # Shuffle Treatment
        cf = CausalForestDML(model_y=RandomForestRegressor(),
                             model_t=RandomForestRegressor(),
                             discrete_treatment=False,
                             random_state=42)
        cf.fit(Y, T_placebo, X=X[model_features])
        placebo_effects.append(np.mean(cf.effect(X[model_features])))
    return placebo_effects
