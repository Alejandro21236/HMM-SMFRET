import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
# === Load and prepare the dataset ===
df = pd.read_csv("/scratch/FRET_true/APO/APO.csv")

# Input and target columns
input_cols = [
    "COM_N_C",
    "Interdomain_Angle",
    "Radius_of_Gyration",
    "Phi_mean",
    "Psi_mean",
    "Torsion_angle",
    "B_factor_HNH",
    "SASA_total",
    "Hydrophobic_SASA",
    "Hydrophilic_SASA",
    "Mean_r",
    "StdDev_r",
    "Kurtosis_r",
    "Autocorrelation_r_lag0",
    "Autocorrelation_r_lag1"
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
# Split sequences before creating tensors
window_size = 20

X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

from sklearn.model_selection import train_test_split

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
)

# Convert to tensors AFTER the split
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# Create DataLoader only for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

time = df["Time_ns"].values[window_size:]
r_true = df["r_34_110"].values[window_size:]

# === Define LSTM model ===
class FRET_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
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
    for batch_X, batch_y in train_loader:
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
initiaf = df["E_34_110"].values[:-1]
finalf= df["E_34_110"].values[1:]

plt.figure(figsize=(6, 5))
plt.hist2d(initiaf, finalf, bins=50, cmap='viridis')
plt.colorbar(label='Counts')
plt.xlabel('Initial FRET')
plt.ylabel('Final FRET')
plt.title('Transition Density Plot (Predicted FRET)')
plt.tight_layout()
plt.savefig("transition_density_plot.png")
plt.show()

n_features = len(input_cols)
fig, axs = plt.subplots(n_features, 1, figsize=(10, 2 * n_features), sharex=True)

# Plot each feature over time
for i, col in enumerate(input_cols):
    axs[i].plot(df["Time_ns"], df[col], label=col, color="blue")
    axs[i].set_ylabel(col)
    axs[i].legend(loc="upper right")
    axs[i].grid(True)

axs[-1].set_xlabel("Time (ns)")
plt.suptitle("Feature Dynamics Over Time", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("feature_dynamics_over_time.png")
plt.show()

import matplotlib.pyplot as plt

# Keywords to exclude
exclude_keywords = ['b_factor', 'mean', 'std', 'min', 'max', 'var', 'skew', 'kurt','autocorrelation']

# Filter input columns
filtered_cols = [col for col in input_cols if not any(kw in col.lower() for kw in exclude_keywords)]

# Create subplots
fig, axs = plt.subplots(len(filtered_cols), 1, figsize=(10, 3 * len(filtered_cols)), sharex=True)

# Ensure axs is always iterable
if len(filtered_cols) == 1:
    axs = [axs]

# Plot each filtered column
for i, col in enumerate(filtered_cols):
    axs[i].plot(df["Time_ns"], df[col], label=col, color="blue")
    axs[i].set_ylabel(col)
    axs[i].legend(loc="upper right")
    axs[i].grid(True)

axs[-1].set_xlabel("Time (ns)")
plt.suptitle("Feature Dynamics Over Time", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("feature_dynamics_over_time.png")
plt.show()
mi_scores = mutual_info_regression(X_scaled, y_scaled.ravel())

# Pair scores with feature names for better interpretation
mi_series = pd.Series(mi_scores, index=input_cols).sort_values(ascending=False)

# Print ranked features by MI score
print(mi_series)
from sklearn.mixture import GaussianMixture
import numpy as np

# Ensure shape is (n_samples, 1)
y_pred = np.array(y_pred).reshape(-1, 1)

# Fit GMM with 3 components (assuming 3 EFRET states)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(y_pred)

# Cluster assignments and probabilities
gmm_labels = gmm.predict(y_pred)
gmm_probs = gmm.predict_proba(y_pred)
import matplotlib.pyplot as plt

x = np.linspace(y_pred.min(), y_pred.max(), 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)

plt.hist(y_pred, bins=100, density=True, alpha=0.4, label='Predicted EFRET Histogram')
plt.plot(x, pdf, label='GMM Fit', linewidth=2)
plt.title("GMM Clustering of Predicted EFRET (y_pred)")
plt.xlabel("EFRET")
plt.ylabel("Density")
plt.legend()
plt.show()
from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(y_pred, gmm_labels)
db_index = davies_bouldin_score(y_pred, gmm_labels)

print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", db_index)
from sklearn.model_selection import KFold
import numpy as np

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_seq)):
    # Split sequences
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    # Initialize model
    model = FRET_LSTM(input_size=X_train.shape[2])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # === Train ===
    for epoch in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    # === Evaluate ===
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().numpy()
        y_true = y_test_tensor.numpy()

        # Rescale to original scale
        y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_true_rescaled = scaler_y.inverse_transform(y_true.reshape(-1, 1))

        # MSE on rescaled outputs
        mse = np.mean((y_true_rescaled - y_pred_rescaled) ** 2)
        mse_scores.append(mse)

        print(f"Fold {fold+1} MSE: {mse:.6f}")

# Final report
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
print(f"\n5-Fold CV MSE: {mean_mse:.6f} ± {std_mse:.6f}")

