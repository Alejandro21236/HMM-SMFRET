import pandas as pd

df = pd.rea_csv("cas9_fret_features")
summary = pd.read_csv("r_t_summary_statistics")
if len summary !=1:
	raise ValuError("the fuck?")
for col in summary:
	frame_features = summary[col].iloc[0]
frame_features_csv = frame_features.to_csv("combined_fret_features.csv", index= False)
print("✅ Combined CSV saved as 'combined_fret_features.csv'")
print(f"Final shape: {frame_features.shape}")
print(f"Included columns: {frame_features.columns.tolist()}") 
