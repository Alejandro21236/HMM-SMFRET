import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
from collections import Counter

# === Load data ===
df = pd.read_csv("RNA.csv")

# === Feature selection ===
feature_cols = [
    "r_34_110", "COM_N_C", "Radius_of_Gyration", "Interdomain_Angle",
    "Phi_mean", "Psi_mean", "Torsion_angle", "SASA_total",
    "Hydrophobic_SASA", "Hydrophilic_SASA", "B_factor_HNH"
]
X = df[feature_cols].values

# === Standardize and reduce dimensionality ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# === HMM training + Viterbi decoding ===
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42, min_covar=1e-3)
model.fit(X_pca)
hidden_states = model.decode(X_pca)[1]

# === State mapping (reverse r correlation) ===
r_index = feature_cols.index("r_34_110")
emission_means = [(i, model.means_[i][r_index]) for i in range(model.n_components)]
sorted_states = sorted(emission_means, key=lambda x: x[1], reverse=True)
state_mapping = {sorted_states[0][0]: 1, sorted_states[1][0]: 0.5, sorted_states[2][0]: 0}
mapped_states = [state_mapping[s] for s in hidden_states]
df["Predicted_FRET_States"] = mapped_states

# === State occupancy ===
counts = Counter(hidden_states)
total = len(hidden_states)
occupancy = {state_mapping[k]: v / total for k, v in counts.items()}

# === Dwell time calculation ===
def get_dwell_times(states, mapping):
    dwell_times = {1: [], 0.5: [], 0: []}
    current = states[0]
    duration = 1
    for s in states[1:]:
        if s == current:
            duration += 1
        else:
            dwell_times[mapping[current]].append(duration)
            current = s
            duration = 1
    dwell_times[mapping[current]].append(duration)
    return dwell_times

dwell_times = get_dwell_times(hidden_states, state_mapping)

# === Save data with predictions ===
df.to_csv("fret_features_with_states.csv", index=False)

# === Plot: Predicted FRET states over time ===
plt.figure(figsize=(10, 4))
plt.plot(df["Time_ns"], df["Predicted_FRET_States"], color="purple", alpha=0.8)
plt.xlabel("Time (ns)")
plt.ylabel("State (0=Inactive, 0.5=Intermediate, 1=Active)")
plt.title("Predicted FRET States Over Time (HMM + Viterbi)")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_fret_states_hmm_viterbi.png")
plt.show()

# === Plot: State occupancy ===
plt.figure(figsize=(5, 4))
plt.bar(occupancy.keys(), occupancy.values(), color="teal")
plt.xlabel("FRET State")
plt.ylabel("Occupancy Fraction")
plt.title("State Occupancy")
plt.xticks([0, 0.5, 1])
plt.tight_layout()
plt.savefig("state_occupancy.png")
plt.show()

# === Plot: Dwell time histograms ===
plt.figure(figsize=(10, 4))
for state, durations in dwell_times.items():
    plt.hist(durations, bins=30, alpha=0.6, label=f"State {state}")
plt.xlabel("Dwell Time (frames)")
plt.ylabel("Frequency")
plt.title("Dwell Time Distributions by State")
plt.legend()
plt.tight_layout()
plt.savefig("dwell_time_histograms.png")
plt.close()
plt.show()
