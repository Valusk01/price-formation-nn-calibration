import os
import numpy as np

from src.datasets.nn_datasets import build_dataset
from src.datasets.nn_datasets import (
    sample_theta_uniform,
    make_sampler_lhs_mixed_delta,
    make_sampler_lhs_box,
)

DTYPE = np.float64

T_SERIES = 2000
N_PATHS  = 50

SIGMA_E_MIN,  SIGMA_E_MAX     = 0.005, 0.10
SIGMA_ETA_MIN, SIGMA_ETA_MAX = 0.005, 0.03
DELTA_MIN,    DELTA_MAX       = 0.05,  0.80

N_TRAIN_PER_DESIGN = 30000
N_TEST_COMMON      = 10000

DATA_DIR = "results/data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_SEED = 12345


def main():
    # Common test set - always uniform in θ
    
    print("\n=== Building common test set (uniform) ===")
    X_test, theta_test, names = build_dataset(
        design_name="test_uniform",
        sampler=sample_theta_uniform,
        n_theta=N_TEST_COMMON,
        T=T_SERIES,
        N_paths=N_PATHS,
        seed=BASE_SEED + 1000,
        sigma_e_min=SIGMA_E_MIN,
        sigma_e_max=SIGMA_E_MAX,
        sigma_eta_min=SIGMA_ETA_MIN,
        sigma_eta_max=SIGMA_ETA_MAX,
        delta_min=DELTA_MIN,
        delta_max=DELTA_MAX,
    )

    test_path = os.path.join(DATA_DIR, "moments_test_uniform.npz")
    np.savez_compressed(
        test_path,
        X=X_test,
        theta=theta_test,
        names=np.array(names, dtype=object),
        T=T_SERIES,
        N_paths=N_PATHS,
        design="test_uniform",
    )
    print("Saved common test set to:", test_path)

    # LHS-based samplers
    lhs_box_sampler = make_sampler_lhs_box(
        n_theta=N_TRAIN_PER_DESIGN,
        sigma_e_min=SIGMA_E_MIN,
        sigma_e_max=SIGMA_E_MAX,
        sigma_eta_min=SIGMA_ETA_MIN,
        sigma_eta_max=SIGMA_ETA_MAX,
        delta_min=DELTA_MIN,
        delta_max=DELTA_MAX,
        seed=BASE_SEED + 888,
    )

    lhs_mixeddelta_sampler = make_sampler_lhs_mixed_delta(
        n_theta=N_TRAIN_PER_DESIGN,
        sigma_e_min=SIGMA_E_MIN,
        sigma_e_max=SIGMA_E_MAX,
        sigma_eta_min=SIGMA_ETA_MIN,
        sigma_eta_max=SIGMA_ETA_MAX,
        delta_min=DELTA_MIN,
        delta_max=DELTA_MAX,
        alpha_tricky=0.7,
        tricky_low=0.575,
        tricky_high=0.725,
        seed=BASE_SEED + 999,
    
    )

    
    # Training datasets for each design
    
    design_samplers = {
        "uniform":        sample_theta_uniform,
        "lhs_box":        lhs_box_sampler,       
        "lhs_mixeddelta": lhs_mixeddelta_sampler,
    }

    for idx, (design_name, sampler) in enumerate(design_samplers.items()):
        print(f"\n=== Building TRAIN dataset for design: {design_name} ===")

        # Seed here affects simulation of time series, not the θ design
        seed = BASE_SEED + 10_000 * (idx + 1)

        X_train, theta_train, names_train = build_dataset(
            design_name=design_name,
            sampler=sampler,
            n_theta=N_TRAIN_PER_DESIGN,
            T=T_SERIES,
            N_paths=N_PATHS,
            seed=seed,
            sigma_e_min=SIGMA_E_MIN,
            sigma_e_max=SIGMA_E_MAX,
            sigma_eta_min=SIGMA_ETA_MIN,
            sigma_eta_max=SIGMA_ETA_MAX,
            delta_min=DELTA_MIN,
            delta_max=DELTA_MAX,
        )

        out_path = os.path.join(DATA_DIR, f"moments_train_{design_name}.npz")
        np.savez_compressed(
            out_path,
            X=X_train,
            theta=theta_train,
            names=np.array(names_train, dtype=object),
            T=T_SERIES,
            N_paths=N_PATHS,
            design=design_name,
        )
        print("Saved training dataset to:", out_path)


if __name__ == "__main__":
    main()
