import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Constants
R0 = 50.0
window_size = 20  # should match your model's training

# Load features
df = pd.read_csv("combined_fret_features.csv")
r_true = df["r_34_110"].values
time = df["Time_ns"].values[window_size:]  # offset for window

# Prepare input for model
input_cols = [
    "COM_N_C", "Interdomain_Angle", "Radius_of_Gyration",
    "Mean_r", "StdDev_r", "Kurtosis_r", 
    "Autocorrelation_r_lag0", "Autocorrelation_r_lag1"
]
X = df[input_cols].values
y = df[["r_34_110"]].values

# Scale inputs and targets
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Sequence batching
def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

# Define model structure (should match your training script)
class FRET_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(FRET_LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load model
model = FRET_LSTM(input_size=X_tensor.shape[2])
model.load_state_dict(torch.load("trained_lstm_model.pth"))  # replace with your model path
model.eval()

# Predict r(t) and convert to efficiency
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

# True r(t) from raw CSV, offset to align with predicted
r_true_trimmed = r_true[window_size:]

# Compute efficiencies
E_true = 1 / (1 + (r_true_trimmed / R0)**6)
E_pred = 1 / (1 + (y_pred.flatten() / R0)**6)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, E_true, label="True Efficiency", color="blue", alpha=0.7)
plt.plot(time, E_pred, label="Predicted Efficiency", color="red", linestyle="--", alpha=0.7)
plt.xlabel("Time (ns)")
plt.ylabel("FRET Efficiency")
plt.title("True vs Predicted FRET Efficiency Over Time (from r(t))")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_vs_time_from_r.png")
plt.show()
