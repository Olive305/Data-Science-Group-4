import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# import data
location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
df_Kundenmonitor2023 = pd.read_excel(location, sheet_name="EE")
location = os.path.join(os.path.dirname(__file__), '../data/custom_files/summary_df_2024.xlsx')
df_Kundenmonitor2024 = pd.read_excel(location)
location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_churn = pd.read_excel(location)

# Print heads for inspection
print(df_Kundenmonitor2023.head())
print(df_Kundenmonitor2024.head())

# Combine Year and Quarter for easier calculations
df_churn['Date'] = pd.to_datetime(df_churn['Jahr'].astype(str) + 'Q' + df_churn['Quartal'].astype(str))

# Calculate the percentage difference in members compared to the year after the current year
df_churn['Mitglieder_diff_next'] = (df_churn.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df_churn['Mitglieder']) / df_churn['Mitglieder']

# Calculate the average churn rate for each insurance company
average_churn = df_churn.groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()

# Create a new DataFrame with the average churn values
df_average_churn = average_churn.rename(columns={'Mitglieder_diff_next': 'Average_Churn_Rate'})

# Add a column to the dataframe with the churn rate in 2023 and 2024
df_churn_2023 = df_churn[df_churn['Jahr'] == 2023].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2023 = df_churn_2023.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2023'})
df_churn_2024 = df_churn[df_churn['Jahr'] == 2024].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2024 = df_churn_2024.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2024'})
df_average_churn = pd.merge(df_average_churn, df_churn_2023, on='Krankenkasse', how='left')
df_average_churn = pd.merge(df_average_churn, df_churn_2024, on='Krankenkasse', how='left')

print(df_average_churn.head)

# Prepare both Kundenmonitor datasets for merging
def prepare_kundenmonitor(df, year):
    df_t = df.set_index('Unnamed: 0').transpose()
    df_t = df_t.reset_index().rename(columns={'index': 'Krankenkasse'})
    df_t['Year'] = year
    # Deduplicate column names robustly
    if hasattr(pd.io.parsers, 'ParserBase'):
        df_t.columns = pd.io.parsers.ParserBase._maybe_dedup_names(list(df_t.columns))
    else:
        # Manual deduplication fallback
        def dedup_columns(cols):
            counts = {}
            new_cols = []
            for col in cols:
                if col not in counts:
                    counts[col] = 0
                    new_cols.append(col)
                else:
                    counts[col] += 1
                    new_cols.append(f"{col}.{counts[col]}")
            return new_cols
        df_t.columns = dedup_columns(list(df_t.columns))
    return df_t

df_Kundenmonitor2023_t = prepare_kundenmonitor(df_Kundenmonitor2023, 2023)
df_Kundenmonitor2024_t = prepare_kundenmonitor(df_Kundenmonitor2024, 2024)

# Align columns before concatenation to avoid InvalidIndexError
common_cols = df_Kundenmonitor2023_t.columns.intersection(df_Kundenmonitor2024_t.columns)
df_Kundenmonitor2023_t = df_Kundenmonitor2023_t[common_cols]
df_Kundenmonitor2024_t = df_Kundenmonitor2024_t[common_cols]

# Concatenate both years
df_kundenmonitor_all = pd.concat([df_Kundenmonitor2023_t, df_Kundenmonitor2024_t], ignore_index=True)
df_kundenmonitor_all = pd.merge(
    df_kundenmonitor_all,
    df_average_churn[['Krankenkasse', 'Churn_Rate_2023', 'Churn_Rate_2024']],
    on='Krankenkasse',
    how='left'
)


# Fill nan values with 2.5
df_kundenmonitor_all = df_kundenmonitor_all.fillna(2.5)

if False:
    # Store this as a excel file
    output_path = os.path.join(os.path.dirname(__file__), '../data/custom_files/kundenmonitor_churn_merged.xlsx')
    df_kundenmonitor_all.to_excel(output_path, index=False)


print(df_kundenmonitor_all.head)

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

# Normalize y values
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

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
epochs = 20
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