import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Load Combined CSV ===
df = pd.read_csv("combined_fret_features.csv")  # this file contains both frame-wise + statistical metrics

# === Specify Inputs and Target ===
input_cols = [
    "E_S867_N1054", "COM_REC1_HNH", "Interdomain_Angle", "Radius_of_Gyration",
    "Mean_r", "StdDev_r", "Kurtosis_r", "Autocorrelation_r_lag0", "Autocorrelation_r_lag1"
]
target_col = "E_S355_S867"

# === Extract Features and Target ===
X = df[input_cols].values
y = df[[target_col]].values
time = df["Time_ps"].values  # for plotting

# === Normalize ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# === Create Sliding Window Sequences ===
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# === Prepare PyTorch DataLoader ===
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === Define LSTM Model ===
class FRET_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FRET_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # only last time step
        return out

model = FRET_LSTM(input_size=X_tensor.shape[2], hidden_size=64, num_layers=2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train Model ===
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
