import pandas as pd
import numpy as np
import matplotlib as plt
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

features_df = pd.read_csv("cas9_fret_features.csv")

E1= features_df["E_S355_S867"].to_numpy()
E2 = features_df["E_S867_N1054"].to_numpy()
R0 = 50
r_t=(1/(E1-1))**(1/6)*R0
mean_r = np.mean(r_t)
std_r = np.std(r_t)
kurtosis_r =  kurtosis(r_t)
autocorr_r = acf(r_t,nlags=40,fft=true)
hist_vals, bin_edges = np.histogram(r_t, bins=50, density = True)
bin_centers = (bin_deges[:-1]+bin_edges[1:])/2
hist_df = pd.dataframe({
	"Bin_Center": bin_centers,
	"Probability": hist_vals
})
hist_df.to_csv("r_t_probability_distribution.csv", index = false)
plt.figure(figsize=(8,5))
plt.hist(E1,bins=50,alpha=.6,label= "E1", color='blue')
plt.hist(E1,bins=50,alpha=.6,label= "E2", color='green')
plt.xlabe("FRET Efficiency")
plt.ylabel("Probability Density")
plt.title("FRET Efficiency Histograms")
plt.show()

#summary 
summary_stats = {
	"mean": mean_r,
	"Stdev_r": std_r,
	"Kurtosis": kurtosis_r,
	"Autocorrelation_rlag0": autocorr_r[0],
	"Autocorrelation_lag1": autocorr_r[1]
}
stats_df = pd.DataFrame([summary_stats])
stats_df.to_csv("r_t_summary_stats.csv", index=False)
print(stats_df)
