import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Load your data
df = pd.read_csv("RNA.csv")

# Define features and target
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
target = "E_34_110"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Store results
results = []

for feature in features:
    X = sm.add_constant(df[[feature]])
    y = df[target]

    model = sm.OLS(y, X).fit()

    # Extract statistics
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    coef = model.params[feature]
    t = model.tvalues[feature]
    p = model.pvalues[feature]
    ci_lower, ci_upper = model.conf_int().loc[feature]
    f_stat = model.fvalue
    f_pval = model.f_pvalue
    corr, corr_p = pearsonr(df[feature], y)

    results.append({
        "Feature": feature,
        "R²": r2,
        "Adjusted R²": adj_r2,
        "Coefficient": coef,
        "t-value": t,
        "p-value": p,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "F-statistic": f_stat,
        "F p-value": f_pval,
        "Pearson r": corr,
        "Pearson p-value": corr_p
    })

# Convert to DataFrame and write to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("feature_statistics_RNA.csv", index=False)
