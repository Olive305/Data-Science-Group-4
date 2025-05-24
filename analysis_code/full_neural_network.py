import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import f1_score

# Get the df with the satisfaction values
file_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
df_sat = pd.read_excel(file_path)

# Switch the krankenkasse names to lowercase and remove spaces
df_sat['Krankenkasse'] = df_sat['Krankenkasse'].str.lower().str.replace(' ', '')

# Get the df with the morbidity rate and the churn rate
df_morb = fuz_combine_fees_morbidity()

# Sort both dataframes by 'Krankenkasse' and 'Jahr' for merge_asof
df_sat = df_sat.sort_values(['Jahr'])
df_morb = df_morb.sort_values(['Jahr'])

# Merge the dataframes using the year and the Krankenkasse values
# Since there are only satisfaciton values for 2023 and 2024, we use the nearest of the two years, when filling the table
df_merged = pd.merge_asof(
    df_morb,
    df_sat,
    on='Jahr',
    by='Krankenkasse',
    direction='nearest'
)

output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/full_data.xlsx')
df_merged.to_excel(output_path, index=False)

# Fill empty values of the df_merged with the mean of the column
df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

# Ensure 'Quartal' is integer for sorting
df_merged['Quartal'] = df_merged['Quartal'].astype(int)

# Sort by 'Krankenkasse', 'Jahr', 'Quartal'
df_merged = df_merged.sort_values(['Krankenkasse', 'Jahr', 'Quartal']).reset_index(drop=True)

# Calculate percentual change in members compared to previous quarter for each Krankenkasse
df_merged['Mitglieder_pct_change_next'] = (
    df_merged.groupby('Krankenkasse')['Mitglieder']
    .pct_change(periods=-1) * 100
)

import torch.nn as nn
import torch.optim as optim

# Select satisfaction columns (skip 'Krankenkasse' and 'Jahr')
satisfaction_columns = df_merged.columns.difference(['Krankenkasse', 'Jahr', 'Churn_Rate_2023', 'Churn_Rate_2024', 'Quartal', 'Regionalität', 'Regionale Verteilung'])

# Remove rows with string values in any satisfaction column
def has_string(row):
    return any(isinstance(row[col], str) for col in satisfaction_columns)

df_merged = df_merged[~df_merged.apply(has_string, axis=1)].reset_index(drop=True)

# Prepare input (satisfaction columns) and output (Mitglieder percentual change for that year) for all years and quartals
X = df_merged[satisfaction_columns].values.astype(float)
y = df_merged['Mitglieder_pct_change_next'].values.reshape(-1, 1).astype(float)

# Drop rows with NaN values in X or y
valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X = X[valid_indices]
y = y[valid_indices]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network with sigmoid activation at the output
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
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
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

# Evaluate the model
def r2_score(preds, targets):
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - ss_res / ss_tot

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    r2 = r2_score(predictions, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    print(f"R^2 Score: {r2.item():.4f}")
    # Binarize predictions and targets: positive change as 1, zero or negative as 0
    preds_bin = (predictions.numpy().flatten() > 0).astype(int)
    targets_bin = (y_test_tensor.numpy().flatten() > 0).astype(int)
    f1 = f1_score(targets_bin, preds_bin)
    print(f"F1 Score: {f1:.4f}")

# Convert predictions to numpy for further analysis if needed
predictions = predictions.numpy().flatten()
y_test_flat = y_test.flatten()

#print('\nPredictions:', predictions)
#print('\nY of train data:\n', y[:5])


