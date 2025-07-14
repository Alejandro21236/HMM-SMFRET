import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.align import alignto
# === Input files ===
topology_file = "/home/aleyva/Downloads/first_frame.pdb"
trajectory_file = "/mnt/mainpool/storage/ALEX/aleyva/CAS9_PREACTIVE_TO_ACTIVE/mdnvt_conc10mM-US-100ns.dcd"
R0 = 50.0  # Förster radius (Å)

# === Load the system ===
u = mda.Universe(topology_file, trajectory_file)

# === Define domains and FRET markers ===
n_lobe = u.select_atoms("resid 1:113")            # REC lobe (REC1+REC2+start of REC3)
linker1 = u.select_atoms("resid 114:140")         # REC → Bridge/PI linker
bridge_helix = u.select_atoms("resid 60:93")      # Overlaps with REC
hnh_domain = u.select_atoms("resid 775:909")      # HNH (mobile cleavage domain)
linker2 = u.select_atoms("resid 770:774")         # Short linker before HNH
ruvc_domain = u.select_atoms("resid 917:1098")    # RuvC cleavage domain
c_lobe = u.select_atoms("resid 770:1098")

# === Corrected Donor/Acceptor based on SER mapping ===
res_donor = u.select_atoms("resid 353 and name CA")    # Donor site (approx. S355)
res_acceptor = u.select_atoms("resid 865 and name CA") # Acceptor site (approx. S867)

# === Data containers ===
features = {
    "Time_ns": [],
    "r_34_110": [],
    "E_34_110": [],
    "COM_N_C": [],
    "Interdomain_Angle": [],
    "Radius_of_Gyration": []
}
r_values = []
print(f"Number of frames in trajectory: {len(u.trajectory)}")

# === Feature extraction ===
for ts in u.trajectory:
    t = ts.frame * 0.01
    r = distance_array(res_donor.positions, res_acceptor.positions)[0][0]
    E = 1 / (1 + (r / R0)**6)

    com_n = n_lobe.center_of_mass()
    com_c = c_lobe.center_of_mass()
    com_dist = np.linalg.norm(com_n - com_c)

    vec1 = res_donor.positions[0] - com_n
    vec2 = res_acceptor.positions[0] - com_c
    angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) /
                   (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))

    rg = linker1.radius_of_gyration()    
    # Save per-frame features
    r_values.append(r)
    features["Time_ns"].append(t)
    features["r_34_110"].append(r)
    features["E_34_110"].append(E)
    features["COM_N_C"].append(com_dist)
    features["Interdomain_Angle"].append(angle)
    features["Radius_of_Gyration"].append(rg)
# === Global statistical features ===
r_array = np.array(r_values)
acf_vals = acf(r_array, nlags=1, fft=True) if len(r_array) > 1 and np.var(r_array) > 1e-8 else [np.nan, np.nan]

stats = {
    "Mean_r": np.mean(r_array),
    "StdDev_r": np.std(r_array),
    "Kurtosis_r": kurtosis(r_array),
    "Autocorrelation_r_lag0": acf_vals[0],
    "Autocorrelation_r_lag1": acf_vals[1],
}

for key, val in stats.items():
    features[key] = [val] * len(r_array)

# === Output ===
df = pd.DataFrame(features)
df.to_csv("/mnt/mainpool/storage/ALEX/aleyva/combined_fret_features_cas92.csv", index=False)
print("✅ Saved to combined_fret_features.csv")
plt.figure(figsize=(10, 5))
plt.plot(df["Time_ns"], df["E_34_110"], label="FRET Efficiency (E_34_110)")
plt.xlabel("Time (ns)")
plt.ylabel("FRET Efficiency")
plt.title("FRET Efficiency Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/mainpool/storage/ALEX/aleyva/fret_efficiency_vs_time.png")
plt.show()

# === Plot: FRET Histogram ===
plt.figure(figsize=(6, 4))
plt.hist(df["E_34_110"], bins=50, color="skyblue", edgecolor="black", density=True)
plt.xlabel("FRET Efficiency")
plt.ylabel("Probability Density")
plt.title("FRET Efficiency Histogram")
plt.tight_layout()
plt.savefig("/mnt/mainpool/storage/ALEX/aleyva/fret_efficiency_histogram.png")
plt.show()

print("✅ Saved CSV and plots to /mnt/mainpool/storage/ALEX/aleyva/")

