import os
import time
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tuning.nn_hyperparams_search import (
    DTYPE,
    grid_search,
)

# -------------------- config --------------------

DATA_DIR = "results/data"
OUT_DIR  = "results/tuning"
os.makedirs(OUT_DIR, exist_ok=True)

DESIGN_NAME = "uniform"   # which training dataset to use

TRAIN_FILE = os.path.join(DATA_DIR, f"moments_train_{DESIGN_NAME}.npz")

RANDOM_SEED = 42

# hyperparameter grid
DEPTH_LIST    = [2, 3, 4]
WIDTH_LIST    = [64, 128, 256]
DROPOUT_LIST  = [0.0, 0.1]
RESIDUAL_LIST = [False, True]
LR_LIST       = [1e-3, 3e-4]
WD_LIST       = [0.0, 1e-4, 1e-3]

LAMBDA_LIST = [
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 2.0),
]

N_EPOCHS   = 300
BATCH_SIZE = 256
PATIENCE   = 30
TRAIN_FRAC = 0.7


def main():
    # ---------- load dataset ----------
    print("Loading training dataset from:", TRAIN_FILE)
    data = np.load(TRAIN_FILE, allow_pickle=True)
    X     = data["X"].astype(DTYPE)      # (n_theta, n_features)
    theta = data["theta"].astype(DTYPE)  # (n_theta, 3)
    names = list(data["names"])

    print("X.shape     =", X.shape)
    print("theta.shape =", theta.shape)
    print("#features   =", len(names))

    # ---------- build hyperparameter grid ----------
    hyper_grid = []
    for depth, width, dropout, residual, lr, wd, lambdas in itertools.product(
        DEPTH_LIST,
        WIDTH_LIST,
        DROPOUT_LIST,
        RESIDUAL_LIST,
        LR_LIST,
        WD_LIST,
        LAMBDA_LIST,
    ):
        cfg = {
            "depth": depth,
            "width": width,
            "dropout": float(dropout),
            "residual": bool(residual),
            "lr": float(lr),
            "weight_decay": float(wd),
            "lambda_e":   float(lambdas[0]),
            "lambda_eta": float(lambdas[1]),
            "lambda_d":   float(lambdas[2]),
        }
        hyper_grid.append(cfg)

    print(f"Total configs to run: {len(hyper_grid)}")

    # ---------- run grid search ----------
    t0 = time.time()
    results, meta = grid_search(
        X=X,
        theta=theta,
        hyper_grid=hyper_grid,
        train_frac=TRAIN_FRAC,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        random_seed=RANDOM_SEED,
    )
    elapsed = time.time() - t0
    print(f"Grid search finished in {elapsed/60:.1f} minutes")

    # ---------- save results ----------
    df = pd.DataFrame(results)

    out_csv = os.path.join(OUT_DIR, f"nn_grid_{DESIGN_NAME}.csv")
    df.to_csv(out_csv, index=False)
    print("Saved grid search results to:", out_csv)

    # ---------- plot validation loss vs configuration index ----------
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # x-axis: configuration index (1..N)
    config_idx = np.arange(1, len(df) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(config_idx, df["best_val_loss"], marker="o", linestyle="-", linewidth=1)
    plt.xlabel("Configuration index")
    plt.ylabel("Best validation loss")
    plt.title(f"NN grid search – validation loss per config ({DESIGN_NAME})")
    plt.grid(True, alpha=0.3)

    fig_path = os.path.join(fig_dir, f"nn_grid_{DESIGN_NAME}_val_loss.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved validation-loss plot to:", fig_path)

    meta_dict = {
        "design_name": DESIGN_NAME,
        "DEPTH_LIST": DEPTH_LIST,
        "WIDTH_LIST": WIDTH_LIST,
        "DROPOUT_LIST": DROPOUT_LIST,
        "RESIDUAL_LIST": RESIDUAL_LIST,
        "LR_LIST": LR_LIST,
        "WD_LIST": WD_LIST,
        "LAMBDA_LIST": LAMBDA_LIST,
        "TRAIN_FRAC": TRAIN_FRAC,
        "N_EPOCHS": N_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "PATIENCE": PATIENCE,
        "RANDOM_SEED": RANDOM_SEED,
        "n_total": meta["n_total"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
    }

    meta_path = os.path.join(OUT_DIR, f"nn_grid_{DESIGN_NAME}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print("Saved meta info to:", meta_path)


if __name__ == "__main__":
    main()