import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

# === Load and prepare the dataset ===
df = pd.read_csv("/mnt/mainpool/storage/ALEX/aleyva/combined_fret_features.csv")

# Input and target columns
input_cols = [
    "COM_N_C", "Interdomain_Angle",
    "Radius_of_Gyration", "Mean_r", "StdDev_r",
    "Kurtosis_r", "Autocorrelation_r_lag0", "Autocorrelation_r_lag1"
]
target_col = "E_34_110"
X = df[input_cols].values
y = df[[target_col]].values

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Sequence generation
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

time = df["Time_ns"].values[window_size:]
r_true = df["r_34_110"].values[window_size:]

# === Define LSTM model ===
class FRET_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super(FRET_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = FRET_LSTM(input_size=X_tensor.shape[2])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train ===
for epoch in range(50):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

# === Predict ===
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_true_rescaled = scaler_y.inverse_transform(y_tensor.numpy())
R0=50.0
# === Plot raw vs predicted histogram ===
plt.figure(figsize=(6, 4))
plt.hist(y_true_rescaled, bins=50, alpha=0.6, label="True", color="skyblue", edgecolor="black", density=True)
plt.hist(y_pred_rescaled, bins=50, alpha=0.6, label="Predicted", color="salmon", edgecolor="black", density=True)
plt.xlabel("FRET Efficiency")
plt.ylabel("Probability Density")
plt.title("True vs Predicted FRET Efficiency Histogram")
plt.legend()
plt.tight_layout()
plt.savefig("fret_hist_comparison.png")
plt.show()
print(y_pred)
# === Calculate error metric ===
mse = np.mean((y_true_rescaled - y_pred_rescaled) ** 2)
print(f"Mean Squared Error: {mse:.6f}")
E_true = 1 / (1 + (r_true / R0)**6)


# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, E_true, label="True FRET Efficiency", color="blue", alpha=0.8)
plt.plot(time, y_pred, label="Predicted FRET Efficiency", color="red", linestyle="--", alpha=0.8)
plt.xlabel("Time (ns)")
plt.ylabel("FRET Efficiency")
plt.title("FRET Efficiency vs Time (True vs Predicted from r(t))")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_over_time.png")
plt.show()
