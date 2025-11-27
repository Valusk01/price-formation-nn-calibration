import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.diagnostics.identification import (
    compute_semi_elasticities,
    jacobian_cond_grid,
)

IN_NPZ = "results/data/moments_diag.npz"

OUT_SEMI_CSV      = "results/data/semi_elasticities_raw.csv"
OUT_SEMI_NORM_CSV = "results/data/semi_elasticities_normalised.csv"
OUT_KAPPA_NPY     = "results/data/jacobian_cond_grid.npy"


def main():
    # load the moments dataset 
    data = np.load(IN_NPZ, allow_pickle=True)

    moments_flat   = data["moments"]        # (n_nodes, M)
    names          = data["names"]          # (M,)
    grid_sigma_e   = data["grid_sigma_e"]   # (E,)
    grid_sigma_eta = data["grid_sigma_eta"] # (H,)
    grid_delta     = data["grid_delta"]     # (D,)

    E = grid_sigma_e.shape[0]
    H = grid_sigma_eta.shape[0]
    D = grid_delta.shape[0]

    n_nodes, M = moments_flat.shape
    assert n_nodes == E * H * D, (
        f"n_nodes={n_nodes} but E*H*D={E*H*D}. "
        "Check the loop order used when building the grid."
    )

    # reshape to (E, H, D, M) 
    moments_grid = moments_flat.reshape(E, H, D, M)

    print("moments_grid.shape =", moments_grid.shape)
    print("M (number of moments) =", M)

    # numerical gradients wrt parameters
    print("\nComputing numerical gradients (central differences)...")

    dM_dse   = np.gradient(moments_grid, grid_sigma_e,   axis=0, edge_order=2)
    dM_dseta = np.gradient(moments_grid, grid_sigma_eta, axis=1, edge_order=2)
    dM_dd    = np.gradient(moments_grid, grid_delta,     axis=2, edge_order=2)

    grads = np.stack([dM_dse, dM_dseta, dM_dd], axis=-1)  # (E, H, D, M, 3)
    print("grads.shape =", grads.shape)

    # semi-elasticities (raw & normalised)
    print("\nComputing semi-elasticities...")

    df_semi, df_semi_norm = compute_semi_elasticities(
        moments_grid=moments_grid,
        grads=grads,
        grid_sigma_e=grid_sigma_e,
        grid_sigma_eta=grid_sigma_eta,
        grid_delta=grid_delta,
        names=names,
    )

    df_semi.to_csv(OUT_SEMI_CSV)
    df_semi_norm.to_csv(OUT_SEMI_NORM_CSV)

    print("Saved raw semi-elasticities to:", OUT_SEMI_CSV)
    print("Saved normalised semi-elasticities to:", OUT_SEMI_NORM_CSV)

    # Jacobian condition numbers on grid 
    print("\nComputing Jacobian condition numbers over the grid...")

    kappa = jacobian_cond_grid(
        grads=grads,
        rtol_small_singular=1e-12,
        moment_names=names,
        verbose=True,   # set True only to debug dropped moments
    )

    np.save(OUT_KAPPA_NPY, kappa)
    print("Saved kappa grid to:", OUT_KAPPA_NPY)

    # summary
    kappa_flat = kappa.reshape(-1)
    finite = np.isfinite(kappa_flat)
    print("\nCondition numbers summary (finite nodes only):")
    print("  #nodes  =", kappa_flat.size)
    print("  #finite =", finite.sum())
    if finite.any():
        print("  median κ =", np.median(kappa_flat[finite]))
        print("  90th pct =", np.percentile(kappa_flat[finite], 90))
        print("  99th pct =", np.percentile(kappa_flat[finite], 99))
    
    # ---------- rank moments by informativeness for each parameter ----------
    os.makedirs("results/tables", exist_ok=True)

    # df_semi_norm has columns: ["sigma_e", "sigma_eta", "delta"]
    rank_sigma_e   = df_semi_norm.sort_values("sigma_e",   ascending=False)
    rank_sigma_eta = df_semi_norm.sort_values("sigma_eta", ascending=False)
    rank_delta     = df_semi_norm.sort_values("delta",     ascending=False)

    rank_sigma_e.to_csv("results/tables/semi_norm_rank_sigma_e.csv")
    rank_sigma_eta.to_csv("results/tables/semi_norm_rank_sigma_eta.csv")
    rank_delta.to_csv("results/tables/semi_norm_rank_delta.csv")

    # ---------- specificity of moments: how parameter-specific are they? ----------
    # df_semi_norm: rows = moments, columns = ["sigma_e", "sigma_eta", "delta"]
    semi_vals = df_semi_norm[["sigma_e", "sigma_eta", "delta"]].values  # (M, 3)
    tiny = 1e-12
    sum_semi = semi_vals.sum(axis=1, keepdims=True) + tiny              # (M, 1)
    shares = semi_vals / sum_semi                                       # (M, 3)

    # full table with all shares (still useful)
    df_shares = df_semi_norm.copy()
    df_shares["share_sigma_e"]   = shares[:, 0]
    df_shares["share_sigma_eta"] = shares[:, 1]
    df_shares["share_delta"]     = shares[:, 2]
    df_shares.to_csv("results/tables/semi_norm_with_shares_full.csv")

    # === three CSVs: informativeness, share, and combined score ===

    # 1) σ_e
    df_sigma_e = pd.DataFrame(index=df_shares.index)
    df_sigma_e["semi_sigma_e"]   = df_shares["sigma_e"]
    df_sigma_e["share_sigma_e"]  = df_shares["share_sigma_e"]
    df_sigma_e["score_sigma_e"]  = df_sigma_e["semi_sigma_e"] * df_sigma_e["share_sigma_e"]
    df_sigma_e = df_sigma_e.sort_values("score_sigma_e", ascending=False)
    df_sigma_e.to_csv("results/tables/semi_sigma_e_with_share_and_score.csv")

    # 2) σ_η
    df_sigma_eta = pd.DataFrame(index=df_shares.index)
    df_sigma_eta["semi_sigma_eta"]  = df_shares["sigma_eta"]
    df_sigma_eta["share_sigma_eta"] = df_shares["share_sigma_eta"]
    df_sigma_eta["score_sigma_eta"] = df_sigma_eta["semi_sigma_eta"] * df_sigma_eta["share_sigma_eta"]
    df_sigma_eta = df_sigma_eta.sort_values("score_sigma_eta", ascending=False)
    df_sigma_eta.to_csv("results/tables/semi_sigma_eta_with_share_and_score.csv")

    # 3) δ
    df_delta = pd.DataFrame(index=df_shares.index)
    df_delta["semi_delta"]   = df_shares["delta"]
    df_delta["share_delta"]  = df_shares["share_delta"]
    df_delta["score_delta"]  = df_delta["semi_delta"] * df_delta["share_delta"]
    df_delta = df_delta.sort_values("score_delta", ascending=False)
    df_delta.to_csv("results/tables/semi_delta_with_share_and_score.csv")


    # ---------- κ(J) by-node table ----------
    # kappa shape: (E, H, D)
    E, H, D = kappa.shape

    # full parameter grid as arrays of same shape
    sigma_e_grid, sigma_eta_grid, delta_grid = np.meshgrid(
        grid_sigma_e, grid_sigma_eta, grid_delta, indexing="ij"
    )

    kappa_flat = kappa.reshape(-1)
    se_flat    = sigma_e_grid.reshape(-1)
    seta_flat  = sigma_eta_grid.reshape(-1)
    d_flat     = delta_grid.reshape(-1)

    # safe log10(κ)
    finite_pos = np.isfinite(kappa_flat) & (kappa_flat > 0)
    log10_kappa = np.full_like(kappa_flat, np.nan, dtype=np.float64)
    log10_kappa[finite_pos] = np.log10(kappa_flat[finite_pos])

    df_kappa = pd.DataFrame(
        {
            "sigma_e": se_flat,
            "sigma_eta": seta_flat,
            "delta": d_flat,
            "kappa": kappa_flat,
            "log10_kappa": log10_kappa,
        }
    )
    df_kappa.to_csv("results/tables/kappa_by_node.csv", index=False)

    # ---------- figures for κ(J) ----------
    os.makedirs("results/figures", exist_ok=True)

    # histogram of log10 κ
    plt.figure()
    plt.hist(log10_kappa[finite_pos], bins=40)
    plt.xlabel(r"$\log_{10} \kappa(J)$")
    plt.ylabel("Count")
    plt.title("Distribution of condition numbers over grid nodes")
    plt.tight_layout()
    plt.savefig("results/figures/kappa_hist_log10.png", dpi=150)
    plt.close()

    # Histogram of κ (no log)
    plt.figure()
    plt.hist(kappa_flat[finite], bins=40)
    plt.xlabel(r"$\kappa(J)$")
    plt.ylabel("Count")
    plt.title("Distribution of condition numbers over grid nodes")
    plt.tight_layout()
    plt.savefig("results/figures/kappa_hist_raw.png", dpi=150)
    plt.close()




if __name__ == "__main__":
    main()