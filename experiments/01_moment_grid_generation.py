import os
import numpy as np
import pandas as pd

from src.diagnostics.moment_grid import generate_moment_dataset

if __name__ == "__main__":
    # simulation settings
    T = 5000
    N_paths = 300
    dtype = np.float64
    rng_seed = 9

    # parameter grids
    grid_sigma_e   = np.arange(0.005, 0.100 + 1e-12, 0.005)   # 20
    grid_sigma_eta = np.geomspace(5e-3, 3e-2, 12)             # 12 (log)
    grid_delta     = np.arange(0.05, 0.80 + 1e-12, 0.075)     # 11

    # generate the dataset
    thetas, moments, names = generate_moment_dataset(
        T=T,
        N_paths=N_paths,
        grid_sigma_e=grid_sigma_e,
        grid_sigma_eta=grid_sigma_eta,
        grid_delta=grid_delta,
        rng_seed=rng_seed,
        dtype=dtype,
    )

    print("n_theta   =", thetas.shape[0])
    print("n_features=", moments.shape[1])
    print("first theta:", thetas[0])
    print("first moments vector shape:", moments[0].shape)

    # save dataset
    OUT_DIR = "results"
    DATA_DIR = os.path.join(OUT_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    npz_path = os.path.join(DATA_DIR, "moments_diag.npz")
    np.savez_compressed(
        npz_path,
        thetas=thetas,
        moments=moments,
        names=np.array(names, dtype=object),
        grid_sigma_e=grid_sigma_e,
        grid_sigma_eta=grid_sigma_eta,
        grid_delta=grid_delta,
        T=np.array(T),
        N_paths=np.array(N_paths),
    )
    print("Saved moment dataset to:", npz_path)






