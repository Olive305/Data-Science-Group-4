import os
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.data_merging_with_satisfaction import merge_churn_with_satisfaction
from analysis_code.predictive_models import reg_morb_fee_churn

MODEL_PATH = "neural_network_model.pt"
SCALER_PATH = "neural_network_scaler.pkl"
CATEGORICAL_COLS_PATH = "neural_network_categorical_cols.pkl"
SATISFACTION_COLS_PATH = "neural_network_satisfaction_cols.pkl"

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)

def train_and_save_neural_network():
    reg_morb_fee_churn()
    df_merged = merge_churn_with_satisfaction()
    categorical_cols = df_merged.select_dtypes(include=['object', 'category']).columns.tolist()
    df_merged = pd.get_dummies(df_merged, columns=categorical_cols, drop_first=True)
    satisfaction_columns = df_merged.columns.difference(['Mitglieder_pct_change_next', 'Mitglieder_diff_next'])
    X_df = df_merged[satisfaction_columns].apply(pd.to_numeric, errors='coerce')
    X_df = X_df.dropna()
    X = X_df.values.astype(float)
    y = df_merged.loc[X_df.index, 'Mitglieder_pct_change_next'].values.reshape(-1, 1).astype(float)
    valid_indices = (
        ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1) &
        ~np.isinf(X).any(axis=1) & ~np.isinf(y).any(axis=1)
    )
    X = X[valid_indices]
    y = y[valid_indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
        y_true = y_test_tensor.numpy().flatten()
        mse_loss = criterion(torch.tensor(y_pred), torch.tensor(y_true)).item()
        # R^2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        # F1 score: binarize by sign (negative vs positive)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin)

        print(
        f"\nTest Results:\n"
        f"- MSE Loss (nn.MSELoss): {mse_loss:.4f}\n"
        f"- R^2 Score: {r2_score:.4f}\n"
        f"- F1 Score (binarized by sign): {f1:.4f}\n"
        "MSE Loss is the mean squared error between predictions and true values.\n"
        "R^2 Score measures how well predictions approximate the real data (1 is perfect).\n"
        "F1 Score is computed after binarizing the regression output by sign (negative vs positive), for interpretability.\n"
        )
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(categorical_cols, CATEGORICAL_COLS_PATH)
    joblib.dump(list(satisfaction_columns), SATISFACTION_COLS_PATH)
    print("Model, scaler, and columns saved.\n\n")

def load_and_predict_neural_network(input_df):
    # Load artifacts
    scaler = joblib.load(SCALER_PATH)
    categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)
    satisfaction_columns = joblib.load(SATISFACTION_COLS_PATH)
    # One-hot encode input_df
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    # Ensure all columns exist
    for col in satisfaction_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[satisfaction_columns]
    X_input = input_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    X_input_scaled = scaler.transform(X_input)
    X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)
    # Load model
    input_size = X_input.shape[1]
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        predictions = model(X_input_tensor).numpy().flatten()
    print(predictions)

def predict_from_excel(excel_file_path):
    """
    Reads an Excel file into a DataFrame and predicts using the trained neural network.
    Returns the predictions as a numpy array.
    """
    df = pd.read_excel(excel_file_path)
    load_and_predict_neural_network(df)


if __name__ == '__main__':
    train_and_save_neural_network()
