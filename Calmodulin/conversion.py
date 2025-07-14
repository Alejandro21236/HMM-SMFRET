import mdtraj as md

# Full paths to your input files
xtc_path = "/mnt/mainpool/storage/ALEX/aleyva/MDR_00001931/processed/trajectory.xtc"
pdb_path = "/mnt/mainpool/storage/ALEX/aleyva/MDR_00001931/processed/structure.pdb"

# Load and convert
traj = md.load(xtc_path, top=pdb_path)
traj.save_dcd("converted.dcd")

print("✅ Conversion complete. Saved as 'converted.dcd'")


