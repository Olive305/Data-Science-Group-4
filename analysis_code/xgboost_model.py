import os
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import binarize

# Extend path and import custom function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity

# Load satisfaction data
file_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
df_sat = pd.read_excel(file_path)

# Normalize Krankenkasse names
df_sat['Krankenkasse'] = df_sat['Krankenkasse'].str.lower().str.replace(' ', '')

# Load morbidity/churn data
df_morb = fuz_combine_fees_morbidity()

# Sort for merge_asof
df_sat = df_sat.sort_values(['Jahr'])
df_morb = df_morb.sort_values(['Jahr'])

# Merge on year and Krankenkasse
df_merged = pd.merge_asof(
    df_morb,
    df_sat,
    on='Jahr',
    by='Krankenkasse',
    direction='nearest'
)

# Save merged data
output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/full_data.xlsx')
df_merged.to_excel(output_path, index=False)

# Fill NaNs with column means
df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

# Remove unwanted rows with strings in numeric columns
satisfaction_columns = df_merged.columns.difference(['Krankenkasse', 'Jahr', 'Churn_Rate_2023', 'Churn_Rate_2024', 'Quartal', 'Regionalität', 'Regionale Verteilung'])

def has_string(row):
    return any(isinstance(row[col], str) for col in satisfaction_columns)

df_merged = df_merged[~df_merged.apply(has_string, axis=1)].reset_index(drop=True)

# Split inputs and targets
X_2023 = df_merged[df_merged['Jahr'] == 2023][satisfaction_columns].values
y_2023 = df_merged[df_merged['Jahr'] == 2023]['Churn_Rate_2023'].values.reshape(-1, 1)
X_2024 = df_merged[df_merged['Jahr'] == 2024][satisfaction_columns].values
y_2024 = df_merged[df_merged['Jahr'] == 2024]['Churn_Rate_2024'].values.reshape(-1, 1)

X = np.vstack([X_2023, X_2024]).astype(float)
y = np.vstack([y_2023, y_2024]).astype(float)

# Remove rows with NaNs
valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train XGB
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train.ravel())

# Predictions
xgb_preds = xgb_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, xgb_preds)
r2 = r2_score(y_test, xgb_preds)

print(f"XGBoost MSE: {mse:.4f}")
print(f"XGBoost R² Score: {r2:.4f}")
# Convert predictions and actual values to binary using a threshold
threshold = y_test.mean()  # Or another appropriate threshold
y_test_binary = binarize(y_test, threshold=threshold)
y_pred_binary = binarize(xgb_preds.reshape(-1, 1), threshold=threshold)

# Calculate F1 score
f1 = f1_score(y_test_binary, y_pred_binary)
print(f"XGBoost F1 Score: {f1:.4f}")

# Get feature importance scores and names
importance_scores = xgb_model.feature_importances_
feature_names = list(satisfaction_columns)

# Sort features by importance
feature_importance = sorted(zip(importance_scores, feature_names), reverse=True)

# Print top 10 features
print("\nTop 10 Most Important Features:")
for importance, name in feature_importance[:10]:
    print(f"{name}: {importance:.4f}")
