import os
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# import pieces from previous head-tuning
from src.tuning.nn_head_search import (
    DTYPE,
    DEVICE,
    StandardScalerNP,
    MLPTrunkHeadModel,
)


# DATA PREP: TRAIN + SELECTION (VALIDATION) DATASET

def prepare_train_and_selection(
    data_dir: str = "results/data",
    train_design_name: str = "uniform",
    selection_design_name: str = "uniform",
) -> Dict[str, Any]:
    """
    Load training dataset moments_train_<design>.npz and
    selection dataset moments_selection_<design>.npz.

    Scale features using the TRAIN set only.
    Transform targets (σ_e, σ_η, δ) as:
        z_e   = log(σ_e)
        z_η   = log(σ_η)
        z_δ   = δ
    Then standardise each z_* using TRAIN set only.

    Returns dict with torch tensors for train & val and the scalers.
    """
    # ---- load training dataset ----
    train_path = os.path.join(data_dir, f"moments_train_{train_design_name}.npz")
    print(f"[multiseed] Loading TRAIN dataset from: {train_path}")
    train_data = np.load(train_path, allow_pickle=True)
    X_train = train_data["X"].astype(DTYPE)        # (n_train, n_features)
    theta_train = train_data["theta"].astype(DTYPE)  # (n_train, 3)
    names = list(train_data["names"])

    # ---- load selection/validation dataset ----
    sel_path = os.path.join(data_dir, f"moments_selection_{selection_design_name}.npz")
    print(f"[multiseed] Loading SELECTION dataset from: {sel_path}")
    sel_data = np.load(sel_path, allow_pickle=True)
    X_val = sel_data["X"].astype(DTYPE)            # (n_sel, n_features)
    theta_val = sel_data["theta"].astype(DTYPE)    # (n_sel, 3)

    # ---- feature scaling: fit on TRAIN only ----
    feat_scaler = StandardScalerNP().fit(X_train)
    X_train_s = feat_scaler.transform(X_train)
    X_val_s   = feat_scaler.transform(X_val)

    # ---- target transform on TRAIN ----
    sigma_e_train   = theta_train[:, 0]
    sigma_eta_train = theta_train[:, 1]
    delta_train     = theta_train[:, 2]

    sigma_e_val   = theta_val[:, 0]
    sigma_eta_val = theta_val[:, 1]
    delta_val     = theta_val[:, 2]

    z_train = np.stack([
        np.log(sigma_e_train),
        np.log(sigma_eta_train),
        delta_train,
    ], axis=1)

    z_val = np.stack([
        np.log(sigma_e_val),
        np.log(sigma_eta_val),
        delta_val,
    ], axis=1)

    target_scaler = StandardScalerNP().fit(z_train)
    z_train_n = target_scaler.transform(z_train)
    z_val_n   = target_scaler.transform(z_val)

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
    X_val_t   = torch.from_numpy(X_val_s.astype(np.float32))
    y_train_t = torch.from_numpy(z_train_n.astype(np.float32))
    y_val_t   = torch.from_numpy(z_val_n.astype(np.float32))

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val":   X_val_t,
        "y_val":   y_val_t,
        "theta_val": theta_val,          # numpy (n_sel, 3) for raw metrics
        "feat_scaler":   feat_scaler,
        "target_scaler": target_scaler,
        "names":         names,
    }


# TRAINING FOR ONE CONFIG + ONE SEED

def train_one_config_once(
    trunk_cfg: Dict[str, Any],
    head_cfg: Dict[str, Any],
    data: Dict[str, Any],
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train one architecture (trunk_cfg + head_cfg) on TRAIN dataset
    and validate on SELECTION dataset.

    Loss is on normalised targets z (StandardScalerNP),
    with λ_d from trunk_cfg["lambda_d"].

    Returns validation MSE on original parameter scale plus best_val_loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = data["X_train"].to(DEVICE)
    y_train = data["y_train"].to(DEVICE)
    X_val   = data["X_val"].to(DEVICE)
    y_val   = data["y_val"].to(DEVICE)

    theta_val      = data["theta_val"]          # numpy (n_sel, 3)
    target_scaler  = data["target_scaler"]

    n_features = X_train.shape[1]

    model = MLPTrunkHeadModel(
        input_dim=n_features,
        trunk_cfg=trunk_cfg,
        head_cfg=head_cfg,
    ).to(DEVICE)

    lambda_d    = float(trunk_cfg["lambda_d"])
    lr          = float(trunk_cfg["lr"])
    weight_decay = float(trunk_cfg["weight_decay"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch    = -1
    best_state    = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)

            mse_e   = mse_loss(pred[:, 0], yb[:, 0])
            mse_eta = mse_loss(pred[:, 1], yb[:, 1])
            mse_d   = mse_loss(pred[:, 2], yb[:, 2])

            loss = mse_e + mse_eta + lambda_d * mse_d
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_accum / max(1, n_batches)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            mse_e_v   = mse_loss(pred_val[:, 0], y_val[:, 0])
            mse_eta_v = mse_loss(pred_val[:, 1], y_val[:, 1])
            mse_d_v   = mse_loss(pred_val[:, 2], y_val[:, 2])
            val_loss  = mse_e_v + mse_eta_v + lambda_d * mse_d_v

        val_loss_float = float(val_loss.item())

        if verbose:
            print(
                f"epoch {epoch:03d} | train_loss={avg_train_loss:.4f} "
                f"| val_loss={val_loss_float:.4f}"
            )

        # early stopping on validation loss
        if val_loss_float < best_val_loss - 1e-6:
            best_val_loss = val_loss_float
            best_epoch    = epoch
            best_state    = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation metrics on ORIGINAL parameter scale
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).cpu().numpy()   # (n_sel, 3)

    # back to z (unnormalised) then to parameters
    z_val_hat   = target_scaler.inverse_transform(pred_val)
    z_val_true  = target_scaler.inverse_transform(data["y_val"].cpu().numpy())

    sigma_e_hat   = np.exp(z_val_hat[:, 0])
    sigma_eta_hat = np.exp(z_val_hat[:, 1])
    delta_hat     = z_val_hat[:, 2]

    sigma_e_true   = theta_val[:, 0]
    sigma_eta_true = theta_val[:, 1]
    delta_true     = theta_val[:, 2]

    val_mse_e   = float(np.mean((sigma_e_hat   - sigma_e_true)   ** 2))
    val_mse_eta = float(np.mean((sigma_eta_hat - sigma_eta_true) ** 2))
    val_mse_d   = float(np.mean((delta_hat     - delta_true)     ** 2))

    result = {
        "best_val_loss": best_val_loss,
        "val_mse_e":     val_mse_e,
        "val_mse_eta":   val_mse_eta,
        "val_mse_d":     val_mse_d,
        "best_epoch":    best_epoch,
    }

    # attach config info
    result.update(trunk_cfg)
    result.update(head_cfg)

    return result