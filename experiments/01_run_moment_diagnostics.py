import numpy as np
import pandas as pd
from src.structural_model.price_formation_model import PriceFormationModel
from src.moments.definitions import realized_moments

T = 1000
N_paths = 10
rng = np.random.default_rng(9)
dtype = np.float64

grid_sigma_e   = np.arange(0.005, 0.100 + 1e-12, 0.005)   # 20
grid_sigma_eta = np.geomspace(5e-3, 3e-2, 12)            # 12 (log)
grid_delta     = np.arange(0.05,  0.80 + 1e-12, 0.075)    # 11

thetas = []
moments = []

for sigma_e in grid_sigma_e:
    for sigma_eta in grid_sigma_eta:
        for d in grid_delta:
            theta = {
                "sigma_e": sigma_e,
                "sigma_eta": sigma_eta,
                "delta": d,
            }
            model = PriceFormationModel(theta)
            sim = model.simulator(T = T, N = N_paths, rng=rng)
            midq = sim["midquote"]
            mret = sim["returns"]

            moms, names = realized_moments(midq, mret, dtype=np.float64)

            # monte carlo average for each parameter triplet (theta)
            moms_theta = moms.mean(axis=0)
            thetas.append([sigma_e, sigma_eta, d])
            moments.append(moms_theta)

thetas = np.array(thetas)
moments = np.vstack(moments)

corr_matrix = np.corrcoef(moments, rowvar=False)
corr_df = pd.DataFrame(corr_matrix, index=names, columns=names)
corr_df.to_csv("moment_correlations.csv")

print("n_theta =", thetas.shape[0])
print("n_features =", moments.shape[1])
print("first theta:", thetas[0])
print("first moments vector shape:", moments[0].shape)
print(corr_df.round(3))  
