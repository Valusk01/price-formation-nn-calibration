import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CSV_PATH = "results/tuning/nn_grid_uniform.csv"
FIG_DIR  = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# LOAD GRID RESULTS
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
config_idx = np.arange(len(df))

best_val = df["best_val_loss"].values

print(f"Loaded {len(df)} configurations.")


# ---------------------------------------------------------------------
# 1) SCATTER – UNSORTED best_val_loss vs config index
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.scatter(config_idx, best_val, s=10)
plt.xlabel("Configuration index")
plt.ylabel("Best validation loss")
plt.title("NN grid search – validation loss per config (uniform)")
plt.tight_layout()

fig_path1 = os.path.join(FIG_DIR, "nn_val_loss_per_config_uniform_scatter_unsorted.png")
plt.savefig(fig_path1, dpi=200)
plt.close()
print("Saved figure:", fig_path1)


# ---------------------------------------------------------------------
# 2) SCATTER – SORTED best_val_loss vs rank
# ---------------------------------------------------------------------

order = np.argsort(best_val)
sorted_best = best_val[order]
rank = np.arange(len(sorted_best))

plt.figure(figsize=(10, 4))
plt.scatter(rank, sorted_best, s=10)
plt.xlabel("Configuration rank (sorted by best_val_loss)")
plt.ylabel("Best validation loss (sorted)")
plt.title("NN grid search – sorted validation loss (uniform)")
plt.tight_layout()

fig_path2 = os.path.join(FIG_DIR, "nn_val_loss_sorted_uniform.png")
plt.savefig(fig_path2, dpi=200)
plt.close()
print("Saved figure:", fig_path2)


# ---------------------------------------------------------------------
# 3) SORTED best_val_loss + val_mse_d on same plot
# ---------------------------------------------------------------------

if "val_mse_d" in df.columns:
    val_mse_d = df["val_mse_d"].values
    sorted_mse_d = val_mse_d[order]

    plt.figure(figsize=(10, 4))
    plt.plot(rank, sorted_best, marker="o", linestyle="", markersize=3, label="best_val_loss")
    plt.plot(rank, sorted_mse_d, linestyle="-", linewidth=1, label="val_mse_d")

    plt.xlabel("Configuration rank (sorted by best_val_loss)")
    plt.ylabel("Loss / MSE")
    plt.title("Sorted best_val_loss and val_mse_d (uniform)")
    plt.legend()
    plt.tight_layout()

    fig_path3 = os.path.join(FIG_DIR, "nn_val_loss_and_mse_d_sorted_uniform.png")
    plt.savefig(fig_path3, dpi=200)
    plt.close()
    print("Saved figure:", fig_path3)
else:
    print("Column 'val_mse_d' not found in CSV, skipping joint plot.")


# ---------------------------------------------------------------------
# 4) PRINT TOP 20 CONFIGS BY best_val_loss
# ---------------------------------------------------------------------

TOP_K = 20
df_sorted = df.sort_values("best_val_loss", ascending=True).reset_index(drop=True)
top = df_sorted.head(TOP_K)

# columns we *try* to show if they exist
cols_preference = [
    "best_val_loss",
    "val_mse_e",
    "val_mse_eta",
    "val_mse_d",
    "depth",
    "width",
    "residual",
    "dropout",
    "weight_decay",
    "lr",
    "lambda_d",
]

cols_to_print = [c for c in cols_preference if c in top.columns]

print(f"\nTop {TOP_K} configs by best_val_loss:")
print(top[cols_to_print].to_string(index=False))

TOP_K = 10
df_sorted = df.sort_values("val_mse_d", ascending=True).reset_index(drop=True)
top = df_sorted.head(TOP_K)

cols_preference = [
    "best_val_loss",
    "val_mse_e",
    "val_mse_eta",
    "val_mse_d",
    "depth",
    "width",
    "residual",
    "dropout",
    "weight_decay",
    "lr",
    "lambda_d",
]

cols_to_print = [c for c in cols_preference if c in top.columns]

print(f"\nTop {TOP_K} configs by val_mse_d:")
print(top[cols_to_print].to_string(index=False))

if __name__ == "__main__":
    # Nothing else; script runs on import as well, but normal use:
    # python -m experiments.05_plot_nn_grid_results
    pass
