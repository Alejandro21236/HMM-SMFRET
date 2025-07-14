import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load your data
# Make sure your DataFrame has these columns + 'fret_efficiency' as the target
df = pd.read_csv("combined_fret_features.csv")

# Select relevant features
features = [
    "Interdomain_Angle",
    "Radius_of_Gyration",
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
