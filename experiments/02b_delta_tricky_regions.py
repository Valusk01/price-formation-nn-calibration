import os
import sys
import numpy as np
import matplotlib.pyplot as plt

print("=== Delta tricky-region experiment ===")

# ============================================================
# 0. PATHS & LOADING
# ============================================================

# Directory of THIS script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Project folders relative to this script (assuming this file is in `experiments/`)
results_dir = os.path.join(script_dir, "..", "results")
data_dir = os.path.join(results_dir, "data")
figures_dir = os.path.join(results_dir, "figures")

cond_path = os.path.join(data_dir, "jacobian_cond_grid.npy")
npz_path  = os.path.join(data_dir, "moments_diag.npz")

print("Script directory :", script_dir)
print("Results directory:", results_dir)
print("Data directory   :", data_dir)
print("Figures directory:", figures_dir)
print("Trying to load κ from:", cond_path)
print("Trying to load grid from:", npz_path)

# Make sure figures directory exists
os.makedirs(figures_dir, exist_ok=True)

# --- Load condition number grid ---
if not os.path.exists(cond_path):
    print("\n[ERROR] jacobian_cond_grid.npy not found at:")
    print("       ", cond_path)
    print("Check that:")
    print("  - 'results/data/jacobian_cond_grid.npy' exists")
    print("  - This script is inside the 'experiments' folder")
    sys.exit(1)

cond_grid = np.load(cond_path)

print("\n=== Loaded jacobian_cond_grid.npy ===")
print("Shape :", cond_grid.shape)
print("Dtype :", cond_grid.dtype)
print("N dim :", cond_grid.ndim)

# --- Load parameter grids so we know actual δ values ---
if not os.path.exists(npz_path):
    print("\n[ERROR] moments_diag.npz not found at:")
    print("       ", npz_path)
    print("Needed to recover grid_delta (actual δ values).")
    sys.exit(1)

data = np.load(npz_path, allow_pickle=True)
grid_sigma_e   = data["grid_sigma_e"]   # (E,)
grid_sigma_eta = data["grid_sigma_eta"] # (H,)
grid_delta     = data["grid_delta"]     # (D,)

print("\n=== Loaded moments_diag.npz grids ===")
print("grid_sigma_e.shape   =", grid_sigma_e.shape)
print("grid_sigma_eta.shape =", grid_sigma_eta.shape)
print("grid_delta.shape     =", grid_delta.shape)

# ============================================================
# 1. CHOOSE DELTA AXIS (WE KNOW IT'S THE LAST: (E,H,D))
# ============================================================

if cond_grid.ndim != 3:
    print("\n[WARNING] Expected κ grid to be 3D (E,H,D). Got ndim =", cond_grid.ndim)

E, H, D = cond_grid.shape
print("\nAssuming shape (E, H, D) with:")
print("  E (sigma_e)   =", E)
print("  H (sigma_eta) =", H)
print("  D (delta)     =", D)

if D != grid_delta.shape[0]:
    print("\n[ERROR] D from κ grid does not match length of grid_delta:")
    print("  D from κ grid =", D)
    print("  len(grid_delta) =", grid_delta.shape[0])
    sys.exit(1)

# delta is the last axis (axis=2), move it to front => (D, E, H)
delta_axis = 2
print("\nUsing delta axis =", delta_axis, "of cond_grid (Python indexing)")

cond_grid_reordered = np.moveaxis(cond_grid, delta_axis, 0)
num_delta = cond_grid_reordered.shape[0]
rest_shape = cond_grid_reordered.shape[1:]

print("Reordered shape (delta first):", cond_grid_reordered.shape)
print("Number of delta points       :", num_delta)
print("Shape of other dimensions    :", rest_shape)

# Use actual δ grid values
delta_values = grid_delta  # shape (D,)
print("\nFirst few delta values:", delta_values[:min(5, num_delta)])

# ============================================================
# 2. GLOBAL STATISTICS (OVER ALL NODES)
# ============================================================

cond_all = cond_grid_reordered.ravel()

global_min = float(np.nanmin(cond_all))
global_max = float(np.nanmax(cond_all))
global_mean = float(np.nanmean(cond_all))
global_median = float(np.nanmedian(cond_all))
p90 = float(np.nanquantile(cond_all, 0.90))
p95 = float(np.nanquantile(cond_all, 0.95))
p99 = float(np.nanquantile(cond_all, 0.99))

print("\n=== Global condition-number stats (including all deltas) ===")
print(f"min      : {global_min:.4g}")
print(f"max      : {global_max:.4g}")
print(f"mean     : {global_mean:.4g}")
print(f"median   : {global_median:.4g}")
print(f"90% tile : {p90:.4g}")
print(f"95% tile : {p95:.4g}")
print(f"99% tile : {p99:.4g}")

# Histogram of all condition numbers
fig = plt.figure(figsize=(7, 4))
plt.hist(cond_all, bins=80)
plt.yscale("log")
plt.xlabel("Jacobian condition number")
plt.ylabel("Count (log scale)")
plt.title("Global distribution of Jacobian condition numbers")
plt.tight_layout()
hist_path = os.path.join(figures_dir, "cond_histogram.png")
fig.savefig(hist_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved global histogram to: {hist_path}")

# ============================================================
# 3. PER-DELTA "TRICKYNESS" METRICS
# ============================================================

# cond_flat[d, :] = all nodes for a given δ_d
cond_flat = cond_grid_reordered.reshape(num_delta, -1)

tricky_threshold = p95

per_delta_max = np.max(cond_flat, axis=1)
per_delta_mean = np.mean(cond_flat, axis=1)
per_delta_median = np.median(cond_flat, axis=1)
per_delta_high_share = np.mean(cond_flat > tricky_threshold, axis=1)

print("\n=== Per-delta summary ===")
print(f"Tricky threshold (95% global quantile): {tricky_threshold:.4g}")
print("For each delta index i / value δ we compute:")
print("  - max condition")
print("  - mean condition")
print("  - median condition")
print("  - share of points above tricky threshold")

# ============================================================
# 4. IDENTIFY TRICKY / EASY DELTA REGIONS
# ============================================================

tricky_score = per_delta_high_share  # share of high-κ nodes

tricky_score_q75 = float(np.quantile(tricky_score, 0.75))
tricky_score_q25 = float(np.quantile(tricky_score, 0.25))

print("\n=== Tricky vs easy deltas based on share of high-condition points ===")
print(f"tricky_score 25% quantile : {tricky_score_q25:.4g}")
print(f"tricky_score 75% quantile : {tricky_score_q75:.4g}")

tricky_mask = tricky_score >= tricky_score_q75
easy_mask   = tricky_score <= tricky_score_q25

# contiguous ranges for tricky deltas
tricky_ranges = []
start = None
for i in range(num_delta):
    if tricky_mask[i]:
        if start is None:
            start = i
    else:
        if start is not None:
            tricky_ranges.append((start, i - 1))
            start = None
if start is not None:
    tricky_ranges.append((start, num_delta - 1))

# contiguous ranges for easy deltas
easy_ranges = []
start = None
for i in range(num_delta):
    if easy_mask[i]:
        if start is None:
            start = i
    else:
        if start is not None:
            easy_ranges.append((start, i - 1))
            start = None
if start is not None:
    easy_ranges.append((start, num_delta - 1))

print("\nTricky δ index/value ranges (75%+ tricky_score):")
if len(tricky_ranges) == 0:
    print("  None found.")
else:
    for (i0, i1) in tricky_ranges:
        d0 = float(delta_values[i0])
        d1 = float(delta_values[i1])
        print(f"  indices [{i0} .. {i1}]  ->  delta in [{d0:.6g} .. {d1:.6g}]")

print("\nEasy δ index/value ranges (25%- tricky_score):")
if len(easy_ranges) == 0:
    print("  None found.")
else:
    for (i0, i1) in easy_ranges:
        d0 = float(delta_values[i0])
        d1 = float(delta_values[i1])
        print(f"  indices [{i0} .. {i1}]  ->  delta in [{d0:.6g} .. {d1:.6g}]")

# Top-k delta points by tricky_score
k = min(10, num_delta)
idx_sorted_tricky = np.argsort(-tricky_score)

print(f"\nTop {k} most tricky δ values (by tricky_score):")
for rank in range(k):
    idx = idx_sorted_tricky[rank]
    dv  = float(delta_values[idx])
    print(
        f"  rank {rank+1:2d}: idx={idx:3d}, delta={dv:.6g} "
        f"| tricky_score={tricky_score[idx]:.4f} "
        f"| max={per_delta_max[idx]:.4g} "
        f"| mean={per_delta_mean[idx]:.4g}"
    )

# ============================================================
# 5. PLOTS VS DELTA (SAVED)
# ============================================================

fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(delta_values, per_delta_max, marker='.')
ax1.set_ylabel("Max κ(J)")
ax1.set_title("Per-delta condition-number summaries vs δ")

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(delta_values, per_delta_mean, marker='.')
ax2.set_ylabel("Mean κ(J)")

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(delta_values, tricky_score, marker='.')
ax3.axhline(tricky_score_q75, linestyle='--', label="75% tricky_score")
ax3.axhline(tricky_score_q25, linestyle='--', label="25% tricky_score")
ax3.set_ylabel("Tricky score (share > 95% tile)")
ax3.set_xlabel("delta (δ)")
ax3.legend(loc="best")

fig.tight_layout()
per_delta_path = os.path.join(figures_dir, "per_delta_summaries.png")
fig.savefig(per_delta_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved per-delta summary plots to: {per_delta_path}")

# ============================================================
# 6. HEATMAPS FOR A FEW SELECTED DELTAS (SAVED)
# ============================================================

print("\n=== Heatmap slices for selected delta values ===")

if cond_grid_reordered.ndim >= 3:
    tricky_idx = idx_sorted_tricky[0]
    idx_sorted_easy = np.argsort(tricky_score)
    easy_idx = idx_sorted_easy[0]
    mid_idx = num_delta // 2

    selected = [
        ("most_tricky", tricky_idx),
        ("easiest", easy_idx),
        ("middle", mid_idx),
    ]

    for label, idx in selected:
        dv = float(delta_values[idx])
        print(f"  Creating heatmap for {label} delta: idx={idx}, δ={dv:.6g}")

        slice_data = cond_grid_reordered[idx]

        # Reduce to 2D if more dimensions remain (take middle slices)
        while slice_data.ndim > 2:
            middle = slice_data.shape[0] // 2
            slice_data = slice_data[middle]

        fig = plt.figure(figsize=(5, 4))
        im = plt.imshow(slice_data, origin="lower", aspect="auto")
        plt.title(f"{label} δ (idx={idx}, δ={dv:.6g})")
        cbar = plt.colorbar(im)
        cbar.set_label("cond")
        plt.xlabel("axis 1")
        plt.ylabel("axis 0")
        fig.tight_layout()

        fname = f"heatmap_{label}_idx_{idx}.png".lower()
        heatmap_path = os.path.join(figures_dir, fname)
        fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved heatmap to: {heatmap_path}")
else:
    print("Not enough dimensions for 2D heatmaps (need delta + at least 2 more dims).")



