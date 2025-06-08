import numpy as np


def sensitivity_analysis(cf_model, X, scale_range=np.linspace(0.9, 1.1, 5)):
    effects = []
    for scale in scale_range:
        X_perturbed = X.copy()
        for col in X.columns:
            X_perturbed[col] *= scale
        effects.append(np.mean(cf_model.effect(X_perturbed)))
    return effects
