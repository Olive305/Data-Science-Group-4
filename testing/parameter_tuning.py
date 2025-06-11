from sklearn.model_selection import RandomizedSearchCV, KFold
from econml.grf import CausalForest
import numpy as np


def tune_causal_forest(X, T, Y, seeds=range(5), n_iter=20, n_splits=3):
    param_dist = {
        "n_estimators": [100, 200, 400, 600],
        "min_samples_leaf": [5, 10, 20],
        "max_depth": [10, 15, 20, None],
        "max_features": ["sqrt", "log2", None]
    }

    best_params_all_seeds = []

    for seed in seeds:
        print(f"Tuning seed {seed}")
        model = CausalForest(random_state=seed, honest=True)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Wrapper für sklearn kompatibel machen
        def fit_predict_score(estimator, X, T, Y, train_index, test_index):
            X_train, T_train, Y_train = X.iloc[train_index], T[train_index], Y[train_index]
            X_test = X.iloc[test_index]
            estimator.fit(X_train, T_train.reshape(-1, 1), Y_train)
            preds = estimator.predict(X_test)
            # Bewertung: z.B. Varianz der geschätzten Effekte (mehr Varianz heißt besser differenzierte CATEs)
            return np.var(preds)

        # Custom scorer für RandomizedSearchCV
        from sklearn.metrics import make_scorer
        def scorer(estimator, X, T, Y):
            # Cross-validate over splits inside RandomizedSearchCV:
            scores = []
            for train_index, test_index in cv.split(X):
                score = fit_predict_score(estimator, X, T, Y, train_index, test_index)
                scores.append(score)
            return np.mean(scores)

        # make_scorer erwartet nur (estimator, X, y), wir packen T und Y in X DataFrame und definieren wrapper unten
        # sklearn leider nicht perfekt kompatibel mit EconML, hier nur als Konzept

        # Alternative: Manuelles Looping wie unten empfohlen, da RandomizedSearchCV schwierig mit econml

        # Einfacher: Manuelles Random Search:
        import random

        best_score = -np.inf
        best_params = None

        for _ in range(n_iter):
            params = {k: random.choice(v) for k, v in param_dist.items()}
            model = CausalForest(
                n_estimators=params["n_estimators"],
                min_samples_leaf=params["min_samples_leaf"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                honest=True,
                random_state=seed
            )
            # Cross-validation
            scores = []
            for train_index, test_index in cv.split(X):
                X_train, T_train, Y_train = X.iloc[train_index], T[train_index], Y[train_index]
                X_test = X.iloc[test_index]
                model.fit(X_train, T_train.reshape(-1, 1), Y_train)
                preds = model.predict(X_test)
                scores.append(np.var(preds))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"Best params for seed {seed}: {best_params} with score {best_score}")
        best_params_all_seeds.append(best_params)

    return best_params_all_seeds
