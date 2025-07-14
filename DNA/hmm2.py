import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
from tqdm import tqdm

# === Load data ===
df = pd.read_csv("DNA.csv")

# === Extended feature set ===
feature_cols = [
    "r_34_110",
    "COM_N_C",
    "Radius_of_Gyration",
    "Interdomain_Angle",
    "Phi_mean",
    "Psi_mean",
    "Torsion_angle",
    "SASA_total",
    "Hydrophobic_SASA",
    "Hydrophilic_SASA",
    "B_factor_HNH"
]

# === Scale features ===
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA ===
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# === HMM training ===
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42, min_covar=1e-3)
model.fit(X_pca)
hidden_states = model.predict(X_pca)

# === State mapping by emission means (r_34_110 index) ===
r_index = feature_cols.index("r_34_110")
emission_means = [(i, model.means_[i][r_index]) for i in range(model.n_components)]
sorted_states = sorted(emission_means, key=lambda x: x[1], reverse = True)
state_mapping = {sorted_states[0][0]: 1, sorted_states[1][0]: 0.5, sorted_states[2][0]: 0}
mapped_states = [state_mapping[s] for s in hidden_states]

# === Save and plot ===
df["Predicted_FRET_States"] = mapped_states
df.to_csv("fret_features_with_states.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["Time_ns"], df["Predicted_FRET_States"], label="Predicted State", color="purple", alpha=0.8)
plt.xlabel("Time (ns)")
plt.ylabel("State (0=Inactive, 0.5=Intermediate, 1=Active)")
plt.title("Predicted FRET States Over Time (HMM)")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_fret_states_hmm.png")
plt.show()
