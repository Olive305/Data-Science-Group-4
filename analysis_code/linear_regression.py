import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import joblib

from analysis_code.cf_honest_trees import prepare_data

def regression_fm_adj_r2(cv=5, model_path="../models/lin_regression_model.pkl"):
    """
    Load data via shared prepare_data, normalize features,
    run k-fold CV with OLS, and report mean R², adj-R², MSE.
    Save final model with constant-added feature names and scaler.
    """
    # Load normalized data and keep feature names
    df, X_raw, X_normalized, T, Y, scaler = prepare_data(
        filepath="../data/fm_dem_sat_merged.xlsx",
        treatment_col='Date',
        outcome_col="Mitglieder_diff_next",
        year_col="Jahr",
        quarter_col="Quartal",
        period_col="Date",
    )
    #treatment_col = Date bcs else it will not use ZB_diff
    #print(X_normalized.columns)
    # Add intercept (constant)
    X = sm.add_constant(X_normalized)
    feature_names = X.columns.tolist()

    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    adj_r2_scores, r2_scores, mse_scores = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        n_train, p_train = X_train.shape
        adj_r2 = 1 - (1 - r2) * (n_train - 1) / (n_train - p_train)
        mse = mean_squared_error(y_test, y_pred)

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        mse_scores.append(mse)

    print(f"OLS (k={cv} folds, adj-R² optimization)")
    print(f"Mean R²:      {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Mean adj-R²:  {np.mean(adj_r2_scores):.4f} ± {np.std(adj_r2_scores):.4f}")
    print(f"Mean MSE:     {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")

    # Fit final model on all data
    model_full = sm.OLS(Y, X).fit()
    print("\nFinal model coefficients and p-values:")
    for name, coef, pval in zip(feature_names, model_full.params, model_full.pvalues):
        print(f"  {name}: {coef:.4f}, p={pval:.2e}")

    # Save model, feature names, and scaler
    feature_names = [col for col in X.columns if col != 'const']
    joblib.dump({
        "model": model_full,
        "features": feature_names,
        "scaler": scaler,
        "add_constant": True
    }, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == '__main__':
    regression_fm_adj_r2(cv=5)
