# experiments/05_nn_heads_tuning.py

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tuning.nn_head_search import (
    DTYPE,
    TRUNK_GRID,
    HEAD_GRID,
    prepare_data,
    train_one_config,
)

# ----------------------------- settings ------------------------------

DATA_DIR   = "results/data"
OUT_DIR    = "results/tuning"
FIG_DIR    = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

VALID_FRAC  = 0.30
BASE_SEED   = 987
BATCH_SIZE  = 256
MAX_EPOCHS  = 300
PATIENCE    = 30

TRAIN_DESIGN_NAME = "uniform"   # moments_train_uniform.npz


# ------------------------------ main ---------------------------------

if __name__ == "__main__":
    # 1) Prepare data
    print("\n[05_nn_heads_tuning] Preparing data…")
    data = prepare_data(
        data_dir=DATA_DIR,
        design_name=TRAIN_DESIGN_NAME,
        valid_frac=VALID_FRAC,
        seed=BASE_SEED,
    )

    # 2) Build combined config grid:
    #    20 trunk configs (10 × descending flag) × len(HEAD_GRID) head configs
    configs = []
    for t_idx, trunk_cfg in enumerate(TRUNK_GRID):
        for h_idx, head_cfg in enumerate(HEAD_GRID):
            cfg = dict(trunk_cfg)
            cfg.update(head_cfg)
            cfg["trunk_index"] = t_idx
            cfg["head_index"]  = h_idx
            configs.append(cfg)

    print(f"[05_nn_heads_tuning] Total configs to train: {len(configs)}")

    # 3) Train each config and collect results
    results = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        trunk_cfg = {
            "depth":           cfg["depth"],
            "width":           cfg["width"],
            "residual":        cfg["residual"],
            "dropout":         cfg["dropout"],
            "weight_decay":    cfg["weight_decay"],
            "lr":              cfg["lr"],
            "lambda_d":        cfg["lambda_d"],
            "trunk_descending": cfg["trunk_descending"],
        }
        head_cfg = {
            "eta_deep":           cfg["eta_deep"],
            "delta_deep":         cfg["delta_deep"],
            "head_hidden_factor": cfg["head_hidden_factor"],
        }

        seed = BASE_SEED + 10_000 * i

        print(
            f"\n[{i+1}/{len(configs)}] "
            f"depth={trunk_cfg['depth']}, width={trunk_cfg['width']}, "
            f"residual={trunk_cfg['residual']}, desc={trunk_cfg['trunk_descending']}, "
            f"eta_deep={head_cfg['eta_deep']}, delta_deep={head_cfg['delta_deep']}, "
            f"hf={head_cfg['head_hidden_factor']}, "
            f"λ_d={trunk_cfg['lambda_d']}, lr={trunk_cfg['lr']}, wd={trunk_cfg['weight_decay']}"
        )

        res = train_one_config(
            trunk_cfg=trunk_cfg,
            head_cfg=head_cfg,
            data=data,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=seed,
            verbose=False,
        )
        res["config_index"] = i
        res["trunk_index"]  = cfg["trunk_index"]
        res["head_index"]   = cfg["head_index"]
        results.append(res)

        elapsed = time.time() - t0
        print(
            f"  -> best_val_loss={res['best_val_loss']:.5f}, "
            f"val_mse_e={res['val_mse_e']:.5f}, "
            f"val_mse_eta={res['val_mse_eta']:.5f}, "
            f"val_mse_d={res['val_mse_d']:.5f} "
            f"(elapsed {elapsed/60.0:.1f} min)"
        )

    # 4) Save results to CSV
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "nn_heads_grid_uniform_eta_delta.csv")
    df.to_csv(out_csv, index=False)
    print("\n[05_nn_heads_tuning] Saved results to:", out_csv)

    # 5) PLOTS
    # (a) configs vs best_val_loss (unsorted)
    plt.figure(figsize=(8, 4))
    x = np.arange(len(df))
    plt.scatter(x, df["best_val_loss"].values, s=10)
    plt.xlabel("Configuration index")
    plt.ylabel("Best validation loss")
    plt.title("Heads tuning (η/δ deep): best_val_loss by configuration")
    plt.tight_layout()
    fig_path_a = os.path.join(FIG_DIR, "nn_heads_eta_delta_val_loss_by_config.png")
    plt.savefig(fig_path_a, dpi=150)
    plt.close()
    print("Saved figure:", fig_path_a)

    # (b) sorted best_val_loss + sorted val_mse_d overlay
    df_sorted = df.sort_values("best_val_loss", ascending=True).reset_index(drop=True)
    x_sorted = np.arange(len(df_sorted))
    y_loss = df_sorted["best_val_loss"].values
    y_d    = df_sorted["val_mse_d"].values

    plt.figure(figsize=(8, 4))
    plt.plot(x_sorted, y_loss, label="best_val_loss")
    plt.plot(x_sorted, y_d,   label="val_mse_d")
    plt.xlabel("Configuration (sorted by best_val_loss)")
    plt.ylabel("Value")
    plt.title("Heads tuning (η/δ deep): best_val_loss and val_mse_d (sorted)")
    plt.legend()
    plt.tight_layout()
    fig_path_b = os.path.join(FIG_DIR, "nn_heads_eta_delta_val_loss_and_val_mse_d_sorted.png")
    plt.savefig(fig_path_b, dpi=150)
    plt.close()
    print("Saved figure:", fig_path_b)

    TUNING_CSV = "results/tuning/nn_heads_grid_uniform_eta_delta.csv"

    df = pd.read_csv(TUNING_CSV)
    print(f"Loaded {len(df)} configurations from: {TUNING_CSV}\n")

    
    # 1) Top 20 configs by best_val_loss
    
    sort_cols = ["best_val_loss", "val_mse_e", "val_mse_eta", "val_mse_d"]
    cfg_cols = [
        "depth", "width", "residual", "trunk_descending",
        "eta_deep", "delta_deep", "head_hidden_factor",
        "lambda_d", "lr", "weight_decay", "best_epoch",
    ]

    df_sorted = df.sort_values("best_val_loss", ascending=True).reset_index(drop=True)
    top20 = df_sorted.head(20)

    print("="*70)
    print("Top 20 configs by best_val_loss:\n")
    print(
        top20[sort_cols + cfg_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print("="*70 + "\n")

    
    # 2) How many of the top 20 use eta_deep / delta_deep
    
    print("Top 20 – counts of eta_deep:")
    print(top20["eta_deep"].value_counts(dropna=False), "\n")

    print("Top 20 – counts of delta_deep:")
    print(top20["delta_deep"].value_counts(dropna=False), "\n")

    print("Top 20 – joint counts (eta_deep, delta_deep):")
    print(top20.groupby(["eta_deep", "delta_deep"]).size(), "\n")

    
    # 3) Average val_mse_d in each (eta_deep, delta_deep) combination
    #    over ALL configs
    print("All configs – mean val_mse_d by (eta_deep, delta_deep):")
    avg_d = (
        df.groupby(["eta_deep", "delta_deep"])["val_mse_d"]
          .mean()
          .reset_index()
          .sort_values("val_mse_d")
    )
    print(avg_d.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("Top 20 – mean val_mse_d by (eta_deep, delta_deep):")
    avg_d_top20 = (
        top20.groupby(["eta_deep", "delta_deep"])["val_mse_d"]
             .mean()
             .reset_index()
             .sort_values("val_mse_d")
    )
    print(avg_d_top20.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
