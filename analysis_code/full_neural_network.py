import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.fee_morb_combination import fuz_combine_fees_morbidity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

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

import torch.nn as nn
import torch.optim as optim

# Select satisfaction columns (skip 'Krankenkasse' and 'Jahr')
satisfaction_columns = df_merged.columns.difference(['Krankenkasse', 'Jahr', 'Churn_Rate_2023', 'Churn_Rate_2024', 'Quartal', 'Regionalität', 'Regionale Verteilung'])

# Remove rows with string values in any satisfaction column
def has_string(row):
    return any(isinstance(row[col], str) for col in satisfaction_columns)

df_merged = df_merged[~df_merged.apply(has_string, axis=1)].reset_index(drop=True)

# Prepare input (satisfaction columns) and output (churn rate for that year)
X_2023 = df_merged[df_merged['Jahr'] == 2023][satisfaction_columns].values
y_2023 = df_merged[df_merged['Jahr'] == 2023]['Churn_Rate_2023'].values.reshape(-1, 1)
X_2024 = df_merged[df_merged['Jahr'] == 2024][satisfaction_columns].values
y_2024 = df_merged[df_merged['Jahr'] == 2024]['Churn_Rate_2024'].values.reshape(-1, 1)

# Stack both years for combined trainingA
# Stack both years for combined training
X = np.vstack([X_2023, X_2024]).astype(float)
y = np.vstack([y_2023, y_2024]).astype(float)

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
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 40
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
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

# Convert predictions to numpy for further analysis if needed
predictions = predictions.numpy().flatten()
y_test_flat = y_test.flatten()


