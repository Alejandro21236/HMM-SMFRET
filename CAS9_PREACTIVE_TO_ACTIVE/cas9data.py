import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis import rms, dihedrals, contacts
import mdtraj as md
from MDAnalysis.analysis import contacts
import math
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.analysis.dihedrals import Ramachandran, Dihedral
from MDAnalysis.analysis.rms import RMSF
from tqdm import tqdm
# === Input files ===
topology_file = "/home/aleyva/Downloads/first_frame.pdb"
trajectory_file = "/mnt/mainpool/storage/ALEX/aleyva/CAS9_PREACTIVE_TO_ACTIVE/mdnvt_conc10mM-US-100ns.dcd"
R0 = 50.0  # Förster radius (Å)

# === Load the system ===
u = mda.Universe(topology_file, trajectory_file)
traj_mdtraj = md.load(trajectory_file,top= topology_file)

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
    "Radius_of_Gyration": [],
    "Phi_mean": [],
    "Psi_mean": [],
    "Torsion_angle": [],
    "B_factor_HNH": [],
    "SASA_total": [],
    "Hydrophobic_SASA": [],
    "Hydrophilic_SASA": []
}
r_values = []
print(f"Number of frames in trajectory: {len(u.trajectory)}")
# B-factor (RMSF) pre-calculation for HNH domain
rmsf_calc = RMSF(hnh_domain).run()
rmsf_hnh = rmsf_calc.results.rmsf
b_factor_hnh = (8 * np.pi ** 2 / 3) * rmsf_hnh ** 2
b_factor_hnh_mean = np.mean(b_factor_hnh)

# SASA Calculation using MDTraj
sasa = md.shrake_rupley(traj_mdtraj, mode='residue')  # shape: (n_frames, n_residues)
sasa_total_per_frame = sasa.sum(axis=1)  # total SASA per frame

# Optional: Define hydrophobic/hydrophilic residues in MDTraj
hydrophobic_resnames = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"]
hydrophilic_resnames = ["SER", "THR", "ASN", "GLN", "TYR", "HIS", "GLU", "ASP", "LYS", "ARG"]

# Get topology info from MDTraj
top = traj_mdtraj.topology
residues = list(top.residues)

rama = Ramachandran(linker1).run()
phi_psi_angles = rama.results.angles  # shape: (n_frames, n_residues, 2)

# 2. Precompute Torsion angles for all frames (set of 4 atoms)
torsion_atoms = u.select_atoms("resid 34 35 36 37 and name CA")
if len(torsion_atoms) == 4:
    torsion_dihedral = Dihedral([torsion_atoms]).run()
    torsion_angles_all = torsion_dihedral.results.angles[:, 0]  # all frames
else:
    torsion_angles_all = [0] * len(u.trajectory)  # fallback

# === Frame-by-frame loop ===
for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
    # Time
    features["Time_ns"].append(i * .02778)

    # Distance between donor and acceptor
    r = res_donor.positions[0] - res_acceptor.center_of_mass()
    r_34_110 = np.linalg.norm(r)
    features["r_34_110"].append(r_34_110)
    r_values.append(r_34_110)

    # FRET efficiency
    R0 = 50.0  # Example Förster radius in Å
    E = 1 / (1 + (r_34_110 / R0) ** 6)
    features["E_34_110"].append(E)

    # COM_N_C
    com_n = n_lobe.center_of_mass()
    com_c = c_lobe.center_of_mass()
    com_distance = np.linalg.norm(com_n - com_c)
    features["COM_N_C"].append(com_distance)

    # Interdomain angle (use vectors between domain centers)
    vec1 = com_c - com_n
    vec2 = hnh_domain.center_of_mass() - com_n
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi
    features["Interdomain_Angle"].append(angle)

    # Radius of Gyration
    rg = u.atoms.radius_of_gyration()
    features["Radius_of_Gyration"].append(rg)

    # Phi/Psi angles from precomputed data
    phi_mean = np.mean(phi_psi_angles[i, :, 0]) if phi_psi_angles.shape[1] > 0 else 0
    psi_mean = np.mean(phi_psi_angles[i, :, 1]) if phi_psi_angles.shape[1] > 0 else 0
    features["Phi_mean"].append(phi_mean)
    features["Psi_mean"].append(psi_mean)

    # Torsion angle from precomputed data
    features["Torsion_angle"].append(torsion_angles_all[i])

    # B-factor of HNH (precomputed mean)
    features["B_factor_HNH"].append(b_factor_hnh_mean)

    # SASA total (from MDTraj)
    features["SASA_total"].append(sasa_total_per_frame[i])

    # Hydrophobic/Hydrophilic SASA per frame
    hydrophobic_sasa = 0
    hydrophilic_sasa = 0
    for j, res in enumerate(residues):
        sasa_value = sasa[i][j]
        if res.name in hydrophobic_resnames:
            hydrophobic_sasa += sasa_value
        elif res.name in hydrophilic_resnames:
            hydrophilic_sasa += sasa_value

    features["Hydrophobic_SASA"].append(hydrophobic_sasa)
    features["Hydrophilic_SASA"].append(hydrophilic_sasa)

print("Feature extraction complete.")

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
df.to_csv("/mnt/mainpool/storage/ALEX/aleyva/combined_fret_features_cas9.csv", index=False)
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

