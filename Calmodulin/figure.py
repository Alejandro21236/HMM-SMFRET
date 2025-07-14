import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.distances import distance_array
import numpy as np
import pandas as pd

trajectory_file = "trajectory.dcd"
topology_file = "4CMP.pdb"   #will need to be cleaned
u= mda.Universe(topology_file, trajectory_file)

#residues and domains from the paper
rec1 = u.select_atoms("resid 56-718")
hnh = u.select_atoms("resid 766-909")

res_S355 = u.select_atoms("resid 355 and name CA")
res_S867 = u.select_atoms("resid 867 and name CA")
res_N1054 = u.select_atoms("resid 1054 and name CA") 

linker_L1 = u.select_atoms("resid 760-765")
linker_L2 =  u.select_atoms("resid 910-915")
flexible_linkers = u.select_atoms("resid 760-765 or resid 910-915")

features = {
	"r_S355_S867": [],
	"r_S867_N1054": [],
	"E_S355_S867": [],
	"E_S867_N1054": [],
	"COM_REC1_HNH": [],
	"Interdomain_Angle": [],
	"Radius_of_Gyration": [],
}

R0=50.0

#trajectory loading
for ts in u.trajectory:
    frame_time_ns = ts.frame * .01
    r1= distabce_array(res_S355.positions, res_S867.positions)
    r2= distance_array(res_S867.positions, res_N1054.positions)
    E1 = 1/(1+(r1/r0)**6)
    E2 = 1/(1+(r2/R0)**6)
    vec1 = res_S355.positions[0] - com1
    vec2 = res_N1054.positions[0] - com2
    angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))

    # Radius of Gyration (combined L1 + L2)
    rg = flexible_linkers.radius_of_gyration()

    # Store
    features.setdefault("Time_ns",[]).append(frame_time_ps)
    features["r_S355_S867"].append(r1)
    features["r_S867_N1054"].append(r2)
    features["E_S355_S867"].append(E1)
    features["E_S867_N1054"].append(E2)
    features["COM_REC1_HNH"].append(com_dist)
    features["Interdomain_Angle"].append(angle)
    features["Radius_of_Gyration"].append(rg)

# === RMSF Calculation (averaged over trajectory) ===
rmsf_calc = RMSF(flexible_linkers).run()
mean_rmsf = np.mean(rmsf_calc.rmsf)
features["RMSF_Mean"] = [mean_rmsf] * len(u.trajectory)

# === OUTPUT ===
features_df = pd.DataFrame(features)
features_df.to_csv("cas9_fret_features.csv", index=False)
print("Feature extraction complete. Output saved to 'cas9_fret_features.csv'.")

