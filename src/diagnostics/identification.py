import numpy as np
import pandas as pd


def compute_semi_elasticities(
    moments_grid: np.ndarray,
    grads: np.ndarray,
    grid_sigma_e: np.ndarray,
    grid_sigma_eta: np.ndarray,
    grid_delta: np.ndarray,
    names,
    tiny: float = 1e-8,
):
    """
    Compute parameter semi-elasticities of moments on the full grid.

    Parameters
    ----------
    moments_grid : (E, H, D, M)
        Moment values at each grid node (already transformed, MC-averaged).
    grads : (E, H, D, M, 3)
        Numerical derivatives of moments w.r.t. parameters (σ_e, σ_η, δ).
    grid_sigma_e   : (E,)
    grid_sigma_eta : (H,)
    grid_delta     : (D,)
    names : list or 1D array of length M
        Moment names.
    tiny : float
        Small constant to avoid division by zero.

    Returns
    -------
    df_semi : DataFrame (M x 3)
        Median |∂f/∂log θ| across the grid for each moment and parameter.
    df_semi_norm : DataFrame (M x 3)
        Same, but normalised by a robust scale (IQR) of the moment.
    """
    E, H, D, M = moments_grid.shape
    assert grads.shape[:4] == (E, H, D, M)
    assert grads.shape[4] == 3

    names = np.array(names, dtype=object)
    assert names.shape[0] == M

    # ------------ build θ grid with correct broadcasting -------------
    # shapes:
    #   se_grid   : (E, 1, 1, 1)
    #   eta_grid  : (1, H, 1, 1)
    #   delta_grid: (1, 1, D, 1)
    se_grid    = grid_sigma_e[:, None, None, None]    # (E, 1, 1, 1)
    eta_grid   = grid_sigma_eta[None, :, None, None]  # (1, H, 1, 1)
    delta_grid = grid_delta[None, None, :, None]      # (1, 1, D, 1)

    # Broadcast all three to the *same* shape (E, H, D, 1),
    # then stack along the last axis → (E, H, D, 1, 3)
    se_b, eta_b, delta_b = np.broadcast_arrays(se_grid, eta_grid, delta_grid)
    theta = np.stack([se_b, eta_b, delta_b], axis=-1)   # (E, H, D, 1, 3)

    # grads is (E,H,D,M,3); theta is (E,H,D,1,3)
    # broadcasting along the M dimension
    grads_log_theta = grads * theta  # (E,H,D,M,3)

    # semi-elasticities: |∂f/∂log θ| = |θ * ∂f/∂θ|
    # take median over all grid nodes for robustness
    semi = np.median(np.abs(grads_log_theta), axis=(0, 1, 2))  # (M,3)

    # ------------ robust scale for normalisation (IQR of |f|) -------------
    F = moments_grid.reshape(-1, M)    # (#nodes, M)
    absF = np.abs(F)

    p25 = np.percentile(absF, 25, axis=0)
    p75 = np.percentile(absF, 75, axis=0)
    iqr = p75 - p25                    # (M,)

    scale = np.maximum(iqr, tiny)
    semi_norm = semi / scale[:, None]  # (M,3)

    # ------------ pack into DataFrames -------------
    cols = ["sigma_e", "sigma_eta", "delta"]

    df_semi = pd.DataFrame(semi, index=names, columns=cols)
    df_semi_norm = pd.DataFrame(semi_norm, index=names, columns=cols)

    return df_semi, df_semi_norm


def jacobian_cond_grid(
    grads: np.ndarray,
    rtol_small_singular: float = 1e-12,
    moment_names=None,
    verbose: bool = False,
):
    """
    Compute Jacobian condition numbers κ(J) at each grid node.

    Parameters
    ----------
    grads : (E, H, D, M, 3)
        Numerical gradients of moments wrt (σ_e, σ_η, δ):
        grads[e,h,d,m,p] = ∂ moment_m / ∂ θ_p.
    rtol_small_singular : float
        Below this smallest singular value we treat κ as ∞ (non-identifiable).
    moment_names : list/array of length M, optional
        Names of the moments (used only for printing).
    verbose : bool
        If True, print which moments are dropped (NaN/all-zero gradient)
        and a small summary at the end.

    Returns
    -------
    kappa : (E, H, D) array
        Condition number κ(J) per grid node.
    """
    E, H, D, M, P = grads.shape
    assert P == 3, "Expecting 3 parameters (σ_e, σ_η, δ)"

    if moment_names is None:
        moment_names = [f"m{j}" for j in range(M)]
    else:
        moment_names = list(moment_names)

    kappa = np.empty((E, H, D), dtype=np.float64)

    # For debugging: how many times each moment is dropped somewhere on the grid
    dropped_counts = np.zeros(M, dtype=int) if verbose else None

    for i in range(E):
        for j in range(H):
            for k in range(D):
                J = grads[i, j, k, :, :]  # (M, 3)

                # rows with any NaN/inf
                mask_bad = ~np.all(np.isfinite(J), axis=1)
                # rows that are (almost) zero in all 3 components
                mask_zero = np.all(np.abs(J) < 1e-15, axis=1)
                # keep only finite & non-zero
                mask_keep = ~(mask_bad | mask_zero)

                if verbose:
                    # record which moments got dropped at this node
                    dropped_idx = np.where(~mask_keep)[0]
                    dropped_counts[dropped_idx] += 1

                J_valid = J[mask_keep, :]  # (M_valid, 3)

                if J_valid.shape[0] < 3:
                    # fewer than 3 informative moments → cannot identify 3 params
                    kappa[i, j, k] = np.inf
                    continue

                try:
                    # singular values s sorted descending
                    _, s, _ = np.linalg.svd(J_valid, full_matrices=False)
                except np.linalg.LinAlgError:
                    kappa[i, j, k] = np.inf
                    continue

                s_min = s[-1]
                s_max = s[0]

                if s_min < rtol_small_singular:
                    kappa[i, j, k] = np.inf
                else:
                    kappa[i, j, k] = float(s_max / s_min)

    # After scanning the grid, print summary of dropped moments
    if verbose and dropped_counts is not None:
        print("\n[Jacobian] Moments dropped at least once (NaN/zero gradients):")
        any_dropped = np.where(dropped_counts > 0)[0]
        if any_dropped.size == 0:
            print("  None – all moments had finite, non-zero gradients everywhere.")
        else:
            for idx in any_dropped:
                print(f"  {moment_names[idx]}: dropped at {dropped_counts[idx]} grid nodes")

    return kappa
