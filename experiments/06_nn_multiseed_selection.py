import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # NEW

from src.tuning.nn_multiseed_selection import (
    DTYPE,
    prepare_train_and_selection,
    train_one_config_once,
)

DATA_DIR   = "results/data"
OUT_DIR    = "results/tuning"
FIG_DIR    = "results/figures"   # NEW

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_DESIGN_NAME      = "uniform"   # moments_train_uniform.npz
SELECTION_DESIGN_NAME  = "uniform"   # moments_selection_uniform.npz

N_SEEDS    = 10
BASE_SEED  = 202_405
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE   = 30

# ----------------- SELECTED 5 CONFIGS HERE -----------------
SELECTED_CONFIGS = [
    dict(depth=2, width=128, residual=False, trunk_descending=False,
         eta_deep=False, delta_deep=True, head_hidden_factor=0.5,
         lambda_d=1.0, lr=3e-4, weight_decay=1e-4),

    dict(depth=3, width=256, residual=True, trunk_descending=False,
         eta_deep=False, delta_deep=False, head_hidden_factor=1.0,
         lambda_d=1.0, lr=3e-4, weight_decay=1e-4),

    dict(depth=2, width=128, residual=False, trunk_descending=True,
         eta_deep=True, delta_deep=False, head_hidden_factor=0.5,
         lambda_d=1.0, lr=3e-4, weight_decay=1e-4),

    dict(depth=2, width=128, residual=False, trunk_descending=False,
         eta_deep=True, delta_deep=True, head_hidden_factor=0.5,
         lambda_d=1.0, lr=3e-4, weight_decay=1e-4),

    dict(depth=4, width=256, residual=True, trunk_descending=True,
         eta_deep=False, delta_deep=False, head_hidden_factor=1.0,
         lambda_d=1.0, lr=3e-3, weight_decay=0),
]
# -------------------------------------------------------------


if __name__ == "__main__":
    # 1) Prepare train + selection data (scalers fitted on train only)
    print("\n[06_nn_multiseed_selection] Preparing data…")
    data = prepare_train_and_selection(
        data_dir=DATA_DIR,
        train_design_name=TRAIN_DESIGN_NAME,
        selection_design_name=SELECTION_DESIGN_NAME,
    )

    # 2) Loop over configs × seeds
    results = []
    t0 = time.time()

    for cfg_idx, cfg in enumerate(SELECTED_CONFIGS):
        print(f"\n=== Config {cfg_idx} ===")
        print(cfg)

        for s in range(N_SEEDS):
            seed = BASE_SEED + 10_000 * cfg_idx + s
            print(f"\n  [cfg {cfg_idx}, seed {s}] training… (global_seed={seed})")

            # split cfg into trunk + head pieces
            trunk_cfg = {
                "depth":           int(cfg["depth"]),
                "width":           int(cfg["width"]),
                "residual":        bool(cfg["residual"]),
                "trunk_descending": bool(cfg["trunk_descending"]),
                "lambda_d":        float(cfg["lambda_d"]),
                "lr":              float(cfg["lr"]),
                "weight_decay":    float(cfg["weight_decay"]),
                "dropout":         0.0,  # you didn't use dropout in these configs
            }
            head_cfg = {
                "head_type": "delta_deep" if cfg["delta_deep"] or cfg["eta_deep"] else "separate",
                # head_type choice: you can change logic here if you want
                "head_hidden_factor": float(cfg["head_hidden_factor"]),
                "eta_deep":   bool(cfg["eta_deep"]),
                "delta_deep": bool(cfg["delta_deep"]),
            }

            res = train_one_config_once(
                trunk_cfg=trunk_cfg,
                head_cfg=head_cfg,
                data=data,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                seed=seed,
                verbose=False,
            )
            res["config_index"] = cfg_idx
            res["seed_index"]   = s
            results.append(res)

            elapsed = time.time() - t0
            print(
                f"    -> best_val_loss={res['best_val_loss']:.5f}, "
                f"val_mse_e={res['val_mse_e']:.5e}, "
                f"val_mse_eta={res['val_mse_eta']:.5e}, "
                f"val_mse_d={res['val_mse_d']:.5e}, "
                f"(elapsed {elapsed/60.0:.1f} min)"
            )

    # 3) Save all runs
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "nn_multiseed_selection_uniform.csv")
    df.to_csv(out_csv, index=False)
    print("\n[06_nn_multiseed_selection] Saved all multi-seed runs to:", out_csv)

    # 4) Aggregate over seeds: mean/std per config
    agg = df.groupby("config_index")[[
        "best_val_loss", "val_mse_e", "val_mse_eta", "val_mse_d"
    ]].agg(["mean", "std"])

    agg_out = os.path.join(OUT_DIR, "nn_multiseed_selection_uniform_agg.csv")
    agg.to_csv(agg_out)
    print("Saved aggregated metrics to:", agg_out)

    print("\nAggregated metrics (per config_index):")
    print(agg)

        # 5) Plot: mean best_val_loss per config with std error bars
    cfg_idx = agg.index.values
    mean_loss = agg[("best_val_loss", "mean")].values
    std_loss  = agg[("best_val_loss", "std")].values

    plt.figure(figsize=(8, 4))
    plt.errorbar(
        cfg_idx,
        mean_loss,
        yerr=std_loss,
        fmt="o",
        capsize=4,
    )
    plt.xlabel("Configuration index")
    plt.ylabel("Best validation loss (mean ± 1 std over seeds)")
    plt.title("Multi-seed robustness: best_val_loss per configuration")
    plt.xticks(cfg_idx)  # to show integer config indices clearly
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "nn_multiseed_best_val_loss_errorbars.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved figure:", fig_path)