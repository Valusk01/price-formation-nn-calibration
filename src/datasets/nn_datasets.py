import time
import numpy as np

from src.structural_model.price_formation_model import PriceFormationModel
from src.moments.definitions import realized_moments

DTYPE = np.float64

# THETA SAMPLERS

def sample_theta_uniform(
    rng: np.random.Generator,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
):
    """
    Baseline design: simple uniform box in (sigma_e, sigma_eta, delta).
    """
    sigma_e = rng.uniform(sigma_e_min, sigma_e_max)
    sigma_eta = rng.uniform(sigma_eta_min, sigma_eta_max)
    delta = rng.uniform(delta_min, delta_max)

    return float(sigma_e), float(sigma_eta), float(delta)


# LHS + dimple-mixture design for delta

def _generate_theta_lhs_mixed_delta(
    rng: np.random.Generator,
    n_samples: int,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
    alpha_tricky: float = 0.7,
    tricky_low: float = 0.575,
    tricky_high: float = 0.725,
):
    """
    Vectorised design generator.

    - σ_e, σ_η:
        Latin Hypercube Sampling (LHS) on log-scale, then exponentiate.
        ⇒ approximately log-uniform within the given bounds.
    - δ:
        Continuous mixture ("dimple") distribution:

            with prob alpha_tricky  -> U[tricky_low, tricky_high]
            with prob 1-alpha_tricky-> U[delta_min, delta_max]

        Trickiness band [tricky_low, tricky_high] comes from
        Jacobian-condition analysis.
    """

    # LHS for (σ_e, σ_η)
    lhs = np.empty((n_samples, 2))
    for j in range(2):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples

    u_se, u_seta = lhs[:, 0], lhs[:, 1]

    log_se_min,   log_se_max   = np.log(sigma_e_min),   np.log(sigma_e_max)
    log_seta_min, log_seta_max = np.log(sigma_eta_min), np.log(sigma_eta_max)

    sigma_e = np.exp(log_se_min   + u_se   * (log_se_max   - log_se_min))
    sigma_eta = np.exp(log_seta_min + u_seta * (log_seta_max - log_seta_min))

    #  Mixture on δ 
    # Make sure tricky band is clipped to [delta_min, delta_max]
    lo = max(delta_min, tricky_low)
    hi = min(delta_max, tricky_high)

    u = rng.random(n_samples)
    delta = np.empty(n_samples, dtype=float)

    if lo < hi:
        mask_tricky = (u < alpha_tricky)
        n_tricky = mask_tricky.sum()
        n_rest = n_samples - n_tricky

        delta[mask_tricky]  = rng.uniform(lo,        hi,        size=n_tricky)
        delta[~mask_tricky] = rng.uniform(delta_min, delta_max, size=n_rest)
    else:
        # fallback: no valid tricky band, just uniform on whole range
        delta[:] = rng.uniform(delta_min, delta_max, size=n_samples)

    theta = np.column_stack([sigma_e, sigma_eta, delta])
    return theta.astype(DTYPE)


def make_sampler_lhs_mixed_delta(
    n_theta: int,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
    alpha_tricky: float = 0.7,
    tricky_low: float = 0.575,
    tricky_high: float = 0.725,
    seed: int = 0,
):
    """
    Factory returning a sampler with the SAME signature that build_dataset expects:

        sampler(rng, sigma_e_min, sigma_e_max,
                     sigma_eta_min, sigma_eta_max,
                     delta_min, delta_max) -> (sigma_e, sigma_eta, delta)

    Internally:
    - Precomputes an LHS + mixed-delta design of size n_theta.
    - The sampler then just walks through this array in order.
    """

    rng_design = np.random.default_rng(seed)
    theta_all = _generate_theta_lhs_mixed_delta(
        rng_design,
        n_samples=n_theta,
        sigma_e_min=sigma_e_min,
        sigma_e_max=sigma_e_max,
        sigma_eta_min=sigma_eta_min,
        sigma_eta_max=sigma_eta_max,
        delta_min=delta_min,
        delta_max=delta_max,
        alpha_tricky=alpha_tricky,
        tricky_low=tricky_low,
        tricky_high=tricky_high,
    )

    idx = {"i": 0}  # mutable index captured by closure

    def sampler(
        _rng: np.random.Generator,
        _sigma_e_min: float,
        _sigma_e_max: float,
        _sigma_eta_min: float,
        _sigma_eta_max: float,
        _delta_min: float,
        _delta_max: float,
    ):
        i = idx["i"]
        if i >= n_theta:
            # wrap around if build_dataset calls more than n_theta times
            idx["i"] = 0
            i = 0

        se, seta, d = theta_all[i]
        idx["i"] += 1
        return float(se), float(seta), float(d)

    return sampler


# Baseline: LHS on *all* parameters (no tricky focus)

def _generate_theta_lhs_box(
    rng: np.random.Generator,
    n_samples: int,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
):
    """
    Full LHS box in (sigma_e, sigma_eta, delta).

    - LHS u ~ [0,1]^3
    - σ_e, σ_η transformed via log-scale (≈ log-uniform)
    - δ transformed linearly to [delta_min, delta_max] (purely neutral)
    """

    # Latin Hypercube on 3 dimensions
    lhs = np.empty((n_samples, 3))
    for j in range(3):
        perm = rng.permutation(n_samples)
        lhs[:, j] = (perm + rng.random(n_samples)) / n_samples

    u_se, u_seta, u_d = lhs.T

    log_se_min,   log_se_max   = np.log(sigma_e_min),   np.log(sigma_e_max)
    log_seta_min, log_seta_max = np.log(sigma_eta_min), np.log(sigma_eta_max)

    sigma_e = np.exp(log_se_min   + u_se   * (log_se_max   - log_se_min))
    sigma_eta = np.exp(log_seta_min + u_seta * (log_seta_max - log_seta_min))

    delta = delta_min + u_d * (delta_max - delta_min)

    theta = np.column_stack([sigma_e, sigma_eta, delta])
    return theta.astype(DTYPE)


def make_sampler_lhs_box(
    n_theta: int,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
    seed: int = 0,
):
    """
    Factory for a 'lhs_box' sampler with the same one-draw signature
    that build_dataset expects.

    Internally precomputes an LHS design of size n_theta over the whole box.
    """

    rng_design = np.random.default_rng(seed)
    theta_all = _generate_theta_lhs_box(
        rng_design,
        n_samples=n_theta,
        sigma_e_min=sigma_e_min,
        sigma_e_max=sigma_e_max,
        sigma_eta_min=sigma_eta_min,
        sigma_eta_max=sigma_eta_max,
        delta_min=delta_min,
        delta_max=delta_max,
    )

    idx = {"i": 0}

    def sampler(
        _rng: np.random.Generator,
        _sigma_e_min: float,
        _sigma_e_max: float,
        _sigma_eta_min: float,
        _sigma_eta_max: float,
        _delta_min: float,
        _delta_max: float,
    ):
        i = idx["i"]
        if i >= n_theta:
            idx["i"] = 0
            i = 0
        se, seta, d = theta_all[i]
        idx["i"] += 1
        return float(se), float(seta), float(d)

    return sampler

# DATASET BUILDER

def build_dataset(
    design_name: str,
    sampler,
    n_theta: int,
    T: int,
    N_paths: int,
    seed: int,
    *,
    sigma_e_min: float,
    sigma_e_max: float,
    sigma_eta_min: float,
    sigma_eta_max: float,
    delta_min: float,
    delta_max: float,
):
    """
    Generate a dataset of (moments, theta) pairs for a given design.

    Parameters
    ----------
    design_name : str
        Just for logging.
    sampler : callable
        Function like sampler(rng, sigma_e_min, ..., delta_max) -> (sigma_e, sigma_eta, delta)
    n_theta : int
        Number of distinct θ points to simulate.
    T : int
        Time–series length per path.
    N_paths : int
        Number of MC paths per θ (used for averaging the moments).
    seed : int
        RNG seed.
    sigma_e_min, sigma_e_max, sigma_eta_min, sigma_eta_max, delta_min, delta_max : float
        Bounds of the parameter box.

    Returns
    -------
    X      : (n_theta, n_features)
        Moment vectors (MC-averaged) per θ.
    thetas : (n_theta, 3)
        Corresponding θ = [sigma_e, sigma_eta, delta].
    names  : list of str
        Feature names (same order as columns of X).
    """
    rng = np.random.default_rng(seed)

    thetas = []
    moments = []
    names = None

    t_start = time.time()
    print(f"\n[build_dataset] design={design_name}, n_theta={n_theta}, T={T}, N_paths={N_paths}")

    for i in range(n_theta):
        sigma_e, sigma_eta, delta = sampler(
            rng,
            sigma_e_min, sigma_e_max,
            sigma_eta_min, sigma_eta_max,
            delta_min, delta_max,
        )

        theta = {
            "sigma_e": sigma_e,
            "sigma_eta": sigma_eta,
            "delta": delta,
        }
        model = PriceFormationModel(theta)

        sim = model.simulator(T=T, N=N_paths, rng=rng)
        midq = sim["midquote"]
        mret = sim["returns"]

        # moments per path: shape (N_paths, n_features)
        X_paths, names_here = realized_moments(midq, mret, dtype=DTYPE)

        if names is None:
            names = list(names_here)
        else:
            # sanity check: names must be identical
            if names_here != names:
                raise RuntimeError("Feature names changed across simulations – bug somewhere.")

        # MC-average over paths → one row per θ
        X_mean = X_paths.mean(axis=0)
        thetas.append([sigma_e, sigma_eta, delta])
        moments.append(X_mean)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  simulated {i+1}/{n_theta} thetas (elapsed {elapsed:.1f}s)")

    thetas = np.array(thetas, dtype=DTYPE)
    X      = np.vstack(moments).astype(DTYPE)

    print(f"[build_dataset] finished design={design_name} in {time.time() - t_start:.1f}s")
    print(f"  X.shape      = {X.shape}")
    print(f"  thetas.shape = {thetas.shape}")

    return X, thetas, names
