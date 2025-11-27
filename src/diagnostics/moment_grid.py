import time
import numpy as np
from src.structural_model.price_formation_model import PriceFormationModel
from src.moments.definitions import realized_moments


def generate_moment_dataset(
    T: int,
    N_paths: int,
    grid_sigma_e: np.ndarray,
    grid_sigma_eta: np.ndarray,
    grid_delta: np.ndarray,
    rng_seed: int = 9,
    dtype=np.float64,
):
    """
    Simulate midquotes on a 3D parameter grid and compute
    Monte Carlo averaged transformed moments for each θ.

    Returns
    -------
    thetas  : (n_theta, 3) array [sigma_e, sigma_eta, delta]
    moments : (n_theta, M)   array of averaged transformed moments
    names   : list of length M with moment names
    """
    rng = np.random.default_rng(rng_seed)

    thetas = []
    moments = []
    names = None  # we’ll fill this from the first call to realized_moments

    # progress bookkeeping
    n_e   = len(grid_sigma_e)
    n_eta = len(grid_sigma_eta)
    n_d   = len(grid_delta)
    total_nodes = n_e * n_eta * n_d

    print(f"Generating moment dataset on grid: "
          f"{n_e}×{n_eta}×{n_d} = {total_nodes} nodes")
    print(f"T = {T}, N_paths = {N_paths}")
    t0 = time.perf_counter()

    node_idx = 0

    for sigma_e in grid_sigma_e:
        for sigma_eta in grid_sigma_eta:
            for d in grid_delta:
                node_idx += 1  
                theta = {
                    "sigma_e": float(sigma_e),
                    "sigma_eta": float(sigma_eta),
                    "delta": float(d),
                }

                model = PriceFormationModel(theta)
                sim = model.simulator(T=T, N=N_paths, rng=rng)
                midq = sim["midquote"]
                mret = sim["returns"]

                # transformed moments + names
                moms, names_local = realized_moments(midq, mret, dtype=dtype)

                # Monte Carlo average across paths
                moms_theta = moms.mean(axis=0)

                if names is None:
                    names = names_local  # set once

                thetas.append([sigma_e, sigma_eta, d])
                moments.append(moms_theta)


                # progress update
                if node_idx % 100 == 0 or node_idx == total_nodes:
                    elapsed = time.perf_counter() - t0
                    frac = node_idx / total_nodes
                    print(
                        f"  [{node_idx}/{total_nodes}] "
                        f"({frac:5.1%}) done, elapsed {elapsed:7.1f}s"
                    )

    thetas = np.array(thetas, dtype=dtype)              # (n_theta, 3)
    moments = np.vstack(moments).astype(dtype)          # (n_theta, M)

    elapsed_total = time.perf_counter() - t0
    print(f"Finished generating dataset. "
          f"Total nodes: {total_nodes}, total time: {elapsed_total:7.1f}s")

    thetas = np.array(thetas, dtype=dtype)
    moments = np.vstack(moments).astype(dtype)

    return thetas, moments, names