import numpy as np
import matplotlib.pyplot as plt
import hmmlearn as hmm
from hmmlearn import hmm
import pandas as pd

df= pd.read_csv("RNA.csv")
feature_cols = ["r_34_110", "COM_N_C", "Radius_of_Gyration"]
feature_cols = feature_cols = [
    "r_34_110",
    "Mean_r",
    "StdDev_r",
    "Kurtosis_r",
    "Autocorrelation_r_lag0",
    "Autocorrelation_r_lag1"
]
X = df[feature_cols].values
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)

model =hmm.GaussianHMM(n_components = 3, covariance_type = 'diag', n_iter=1000, random_state=42, min_covar=1e-3)
model.fit(X_scaled)
hidden_states= model.predict(X_scaled)
state_means = []
for i in range(model.n_components): 
	state_mean_r =np.mean(df["r_34_110"].values[hidden_states == i])
	state_means.append((i,state_mean_r))
sorted_states=sorted(state_means, key=lambda x: x[1])
state_mapping = {sorted_states[0][0]:1, sorted_states[1][0]: .5, sorted_states[2][0]:0}
mapped_states = [state_mapping[s] for s in hidden_states]
df["Predicted_FRET_States"]=mapped_states

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
