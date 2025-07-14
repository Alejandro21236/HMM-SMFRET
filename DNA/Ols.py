import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load your data
# Make sure your DataFrame has these columns + 'fret_efficiency' as the target
df = pd.read_csv("DNA.csv")

# Select relevant features
features = [
    "SASA_total",
    "Hydrophobic_SASA",
    "Hydrophilic_SASA",
    "Interdomain_Angle",
    "Radius_of_Gyration",
    "Phi_mean",
    "Psi_mean",
    "Torsion_angle",
    "COM_N_C"
]

X = df[features]
y = df['E_34_110']

# Standardize the feature matrix
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)

# Add intercept term
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Display summary statistics (including p-values)
print(model.summary())
