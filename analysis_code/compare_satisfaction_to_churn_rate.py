import os
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# import data
location = os.path.join(os.path.dirname(__file__), '../data/Kundenmonitor_GKV_2023.xlsx')
df_Kundenmonitor = pd.read_excel(location, sheet_name="EE")
location = os.path.join(os.path.dirname(__file__), '../data/Zusatzbeitrag_je Kasse je Quartal.xlsx')
df_churn = pd.read_excel(location)

print(df_Kundenmonitor.head)

#combine Year and Quarter for easier calculations
df_churn['Date'] = pd.to_datetime(df_churn['Jahr'].astype(str) + 'Q'+ df_churn['Quartal'].astype(str))

#calculate the percentage difference in members compared to the year after the current year
df_churn['Mitglieder_diff_next'] = (df_churn.groupby('Krankenkasse')['Mitglieder'].shift(-1) - df_churn['Mitglieder']) / df_churn['Mitglieder']

# Calculate the average churn rate for each insurance company
average_churn = df_churn.groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()

# Create a new DataFrame with the average churn values
df_average_churn = average_churn.rename(columns={'Mitglieder_diff_next': 'Average_Churn_Rate'})

# Add a column to the dataframe with the churn rate in 2023
df_churn_2024 = df_churn[df_churn['Jahr'] == 2024].groupby('Krankenkasse')['Mitglieder_diff_next'].mean().reset_index()
df_churn_2024 = df_churn_2024.rename(columns={'Mitglieder_diff_next': 'Churn_Rate_2024'})
df_average_churn = pd.merge(df_average_churn, df_churn_2024, on='Krankenkasse', how='left')

# Now we add the values of the df_Kundenmonitor into the df_average_churn dataframe such that same insurance companys are together
# Merge the satisfaction data from df_Kundenmonitor into df_average_churn
# Transpose df_Kundenmonitor for easier merging
df_Kundenmonitor_transposed = df_Kundenmonitor.set_index('Unnamed: 0').transpose()

# Reset index and rename columns for merging
df_Kundenmonitor_transposed = df_Kundenmonitor_transposed.reset_index().rename(columns={'index': 'Krankenkasse'})

# Merge the transposed Kundenmonitor data into df_average_churn
df_average_churn = pd.merge(df_average_churn, df_Kundenmonitor_transposed, on='Krankenkasse', how='left')

# Now we try to train a neural network based on the data
# Iterate through each satisfaction column and compare it with the churn rate
satisfaction_columns = df_Kundenmonitor_transposed.columns[3:]

# Prepare the input (satisfaction columns) and output (Churn_Rate_2024)
X = df_average_churn[satisfaction_columns].values
y = df_average_churn['Churn_Rate_2024'].values.reshape(-1, 1)

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

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

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