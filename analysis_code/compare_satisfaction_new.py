import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_churn = pd.read_excel(location)
location = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
df_merged = pd.read_excel(location)

# Combine Year and Quarter for easier calculations
df_churn['Date'] = pd.to_datetime(df_churn['Jahr'].astype(str) + 'Q' + df_churn['Quartal'].astype(str))

# Calculate the percentage difference in members compared to the year after the current year
df_churn['Mitglieder_diff_next'] = (df_churn.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df_churn['Mitglieder']) / df_churn['Mitglieder']

# Calculate the average churn rate for each insurance company
average_churn = df_churn.groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()

# Create a new DataFrame with the average churn values
df_average_churn = average_churn.rename(columns={'Mitglieder_diff_next': 'Average_Churn_Rate'})

# Use correct Krankenkasse names in the second part also

# Add a column to the dataframe with the churn rate in 2023 and 2024
df_churn_2023 = df_churn[df_churn['Jahr'] == 2023].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2023 = df_churn_2023.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2023'})
df_churn_2024 = df_churn[df_churn['Jahr'] == 2024].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2024 = df_churn_2024.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2024'})

# Add these values in the df_merged dataframe
df_merged = df_merged.merge(df_churn_2023, on='Krankenkasse', how='left', suffixes=('', '_new'))
df_merged = df_merged.merge(df_churn_2024, on='Krankenkasse', how='left', suffixes=('', '_new'))

df_merged['Churn_Rate_2023'] = df_merged['Churn_Rate_2023_new'].combine_first(df_merged['Churn_Rate_2023'])
df_merged['Churn_Rate_2024'] = df_merged['Churn_Rate_2024_new'].combine_first(df_merged['Churn_Rate_2024'])

df_merged = df_merged.drop(columns=['Churn_Rate_2023_new', 'Churn_Rate_2024_new'])

if False:
    output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
    df_merged.to_excel(output_path, index=False)

# Check correlation of all features with churn rates
correlation_matrix = df_merged.corr(numeric_only=True)
print("Correlation with Churn_Rate_2023:")
print(correlation_matrix['Churn_Rate_2023'].sort_values(ascending=False))
print("\nCorrelation with Churn_Rate_2024:")
print(correlation_matrix['Churn_Rate_2024'].sort_values(ascending=False))

df_kundenmonitor_all = df_merged

# Select satisfaction columns (skip 'Krankenkasse' and 'Year')
satisfaction_columns = df_kundenmonitor_all.columns.difference(['Krankenkasse', 'Year', 'Churn_Rate_2023', 'Churn_Rate_2024'])

# Prepare input (satisfaction columns) and output (churn rate for that year)
X_2023 = df_kundenmonitor_all[df_kundenmonitor_all['Year'] == 2023][satisfaction_columns].values
y_2023 = df_kundenmonitor_all[df_kundenmonitor_all['Year'] == 2023]['Churn_Rate_2023'].values.reshape(-1, 1)
X_2024 = df_kundenmonitor_all[df_kundenmonitor_all['Year'] == 2024][satisfaction_columns].values
y_2024 = df_kundenmonitor_all[df_kundenmonitor_all['Year'] == 2024]['Churn_Rate_2024'].values.reshape(-1, 1)

# Stack both years for combined training
X = np.vstack([X_2023, X_2024])
y = np.vstack([y_2023, y_2024])

# Drop rows with NaN values in X or y
valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# Scale inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the normalized values
print("Normalized y_train:", y_train.flatten())
print("Normalized y_test:", y_test.flatten())

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

            nn.Linear(32, 1),  # Output for regression
            nn.Sigmoid()       # Sigmoid activation at the output
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model, loss function, and optimizer
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

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss for every epoch
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

# Convert predictions to numpy for further analysis if needed
predictions = predictions.numpy()
print(predictions)