import os
import numpy as np

from src.datasets.nn_datasets import (
    build_dataset,
    sample_theta_uniform,
)

DTYPE = np.float64

T_SERIES = 2000
N_PATHS  = 50

SIGMA_E_MIN,  SIGMA_E_MAX    = 0.005, 0.10
SIGMA_ETA_MIN, SIGMA_ETA_MAX = 0.005, 0.03
DELTA_MIN,    DELTA_MAX      = 0.05, 0.80

N_SELECTION_PER_DESIGN = 10_000

DATA_DIR = "results/data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_SEED = 12345


def main():

    design_samplers = {
        "uniform": sample_theta_uniform,
    }

    for idx, (design_name, sampler) in enumerate(design_samplers.items()):
        print(f"\n=== Building SELECTION/VALIDATION dataset for design: {design_name} ===")

        # new seed
        selection_seed = BASE_SEED + 20_000 * (idx + 1)

        X_sel, theta_sel, names_sel = build_dataset(
            design_name=f"{design_name}_selection",
            sampler=sampler,
            n_theta=N_SELECTION_PER_DESIGN,
            T=T_SERIES,
            N_paths=N_PATHS,
            seed=selection_seed,
            sigma_e_min=SIGMA_E_MIN,
            sigma_e_max=SIGMA_E_MAX,
            sigma_eta_min=SIGMA_ETA_MIN,
            sigma_eta_max=SIGMA_ETA_MAX,
            delta_min=DELTA_MIN,
            delta_max=DELTA_MAX,
        )

        sel_path = os.path.join(DATA_DIR, f"moments_selection_{design_name}.npz")
        np.savez_compressed(
            sel_path,
            X=X_sel,
            theta=theta_sel,
            names=np.array(names_sel, dtype=object),
            T=T_SERIES,
            N_paths=N_PATHS,
            design=f"{design_name}_selection",
        )
        print("Saved selection/validation dataset to:", sel_path)


if __name__ == "__main__":
    main()
