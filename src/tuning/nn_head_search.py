import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DTYPE = np.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# BASE TRUNK CONFIGS (Selected from the first fine tuning)

_df_trunks = pd.DataFrame([
    dict(best_val_loss=0.048039, val_mse_e=0.001219, val_mse_eta=0.021823, val_mse_d=0.024997,
         depth=3, width=128, residual=True,  dropout=0.0, weight_decay=1e-4, lr=1e-3, lambda_d=1.0),

    dict(best_val_loss=0.048349, val_mse_e=0.001105, val_mse_eta=0.021231, val_mse_d=0.026013,
         depth=3, width=64,  residual=True,  dropout=0.0, weight_decay=1e-4, lr=3e-4, lambda_d=1.0),

    dict(best_val_loss=0.048500, val_mse_e=0.000812, val_mse_eta=0.021530, val_mse_d=0.026159,
         depth=2, width=128, residual=False, dropout=0.0, weight_decay=1e-4, lr=3e-4, lambda_d=1.0),

    dict(best_val_loss=0.048887, val_mse_e=0.000779, val_mse_eta=0.021487, val_mse_d=0.026620,
         depth=2, width=128, residual=True,  dropout=0.0, weight_decay=1e-4, lr=1e-3, lambda_d=1.0),

    dict(best_val_loss=0.049245, val_mse_e=0.000699, val_mse_eta=0.022347, val_mse_d=0.026199,
         depth=4, width=256, residual=True,  dropout=0.0, weight_decay=0.0, lr=3e-4, lambda_d=1.0),

    dict(best_val_loss=0.049066, val_mse_e=0.000780, val_mse_eta=0.021497, val_mse_d=0.026789,
         depth=3, width=256, residual=True,  dropout=0.0, weight_decay=1e-4, lr=3e-4, lambda_d=1.0),

    dict(best_val_loss=0.074704, val_mse_e=0.001496, val_mse_eta=0.024105, val_mse_d=0.024552,
         depth=4, width=128, residual=False, dropout=0.0, weight_decay=1e-4, lr=1e-3, lambda_d=2.0),

    dict(best_val_loss=0.075281, val_mse_e=0.001489, val_mse_eta=0.024411, val_mse_d=0.024691,
         depth=4, width=256, residual=False, dropout=0.0, weight_decay=1e-3, lr=3e-4, lambda_d=2.0),

    dict(best_val_loss=0.072825, val_mse_e=0.001126, val_mse_eta=0.022130, val_mse_d=0.024784,
         depth=4, width=128, residual=False, dropout=0.0, weight_decay=1e-4, lr=3e-4, lambda_d=2.0),

    dict(best_val_loss=0.074821, val_mse_e=0.002148, val_mse_eta=0.022748, val_mse_d=0.024963,
         depth=3, width=256, residual=False, dropout=0.0, weight_decay=0.0, lr=1e-3, lambda_d=2.0),
])

TRUNK_GRID: List[Dict[str, Any]] = []
for _, row in _df_trunks.iterrows():
    base = dict(
        depth        = int(row["depth"]),
        width        = int(row["width"]),
        residual     = bool(row["residual"]),
        dropout      = float(row["dropout"]),
        weight_decay = float(row["weight_decay"]),
        lr           = float(row["lr"]),
        lambda_d     = float(row["lambda_d"]),
    )
    for desc in [False, True]:
        cfg = base.copy()
        cfg["trunk_descending"] = desc
        TRUNK_GRID.append(cfg)

print(f"[nn_head_search] trunk configs: {len(TRUNK_GRID)} (10 × 2 descending flags)")


# HEAD CONFIG GRID: ETA / DELTA DEEP FLAGS

HEAD_HIDDEN_FACTORS = [0.5, 1.0]

# Each head config decides:
#   - eta_deep:   False/True (σ_η shallow vs deep 2-layer head)
#   - delta_deep: False/True (δ  shallow vs deep 2-layer head)
#   - head_hidden_factor: multiplier for hidden width (for all deep heads)
HEAD_GRID: List[Dict[str, Any]] = []
for eta_deep in [False, True]:
    for delta_deep in [False, True]:
        for hf in HEAD_HIDDEN_FACTORS:
            HEAD_GRID.append(
                dict(
                    eta_deep=eta_deep,
                    delta_deep=delta_deep,
                    head_hidden_factor=hf,
                )
            )

print(f"[nn_head_search] head configs: {len(HEAD_GRID)} (4 combinations × {len(HEAD_HIDDEN_FACTORS)} factors)")


# DATA LOADING + SCALING

class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std_ + self.mean_


def load_train_dataset(
    data_dir: str = "results/data",
    design_name: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = os.path.join(data_dir, f"moments_train_{design_name}.npz")
    print(f"[nn_head_search] Loading train dataset from: {path}")
    data = np.load(path, allow_pickle=True)

    X = data["X"].astype(DTYPE)
    theta = data["theta"].astype(DTYPE)
    names = list(data["names"])
    return X, theta, names


def prepare_data(
    data_dir: str = "results/data",
    design_name: str = "uniform",
    valid_frac: float = 0.15,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Load dataset, split train/val, scale X and y.

    Targets:
      z_e   = log(sigma_e)
      z_eta = log(sigma_eta)
      z_d   = delta
    Then standardise each z_* on the train set.
    """
    X, theta, names = load_train_dataset(data_dir=data_dir, design_name=design_name)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_valid = int(np.round(valid_frac * n))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    X_train = X[train_idx]
    X_val   = X[valid_idx]
    theta_train = theta[train_idx]
    theta_val   = theta[valid_idx]

    # feature scaling
    feat_scaler = StandardScalerNP().fit(X_train)
    X_train_s = feat_scaler.transform(X_train)
    X_val_s   = feat_scaler.transform(X_val)

    # target transform
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
        "theta_val": theta_val,
        "feat_scaler":   feat_scaler,
        "target_scaler": target_scaler,
        "names":         names,
    }



# MODEL: SHARED TRUNK + FLEXIBLE HEADS

class MLPTrunkHeadModel(nn.Module):
    """
    Shared trunk + flexible heads.

    trunk_cfg:
      - depth, width, residual, dropout, trunk_descending

    head_cfg:
      - eta_deep: bool
      - delta_deep: bool
      - head_hidden_factor: float
    """
    def __init__(
        self,
        input_dim: int,
        trunk_cfg: Dict[str, Any],
        head_cfg: Dict[str, Any],
    ):
        super().__init__()

        depth        = trunk_cfg["depth"]
        width        = trunk_cfg["width"]
        residual     = trunk_cfg["residual"]
        dropout_p    = trunk_cfg["dropout"]
        descending   = trunk_cfg["trunk_descending"]

        self.residual = residual
        self.input_dim = input_dim

        # ----- trunk -----
        layers = []
        in_dim = input_dim

        if depth <= 0:
            self.trunk = nn.Identity()
            trunk_out_dim = input_dim
        else:
            current_width = width
            for _ in range(depth):
                out_dim = current_width
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(dropout_p))
                in_dim = out_dim
                if descending:
                    current_width = max(8, current_width // 2)

            self.trunk = nn.Sequential(*layers)
            trunk_out_dim = in_dim

        self.trunk_out_dim = trunk_out_dim

        # ----- heads -----
        eta_deep   = bool(head_cfg["eta_deep"])
        delta_deep = bool(head_cfg["delta_deep"])
        head_hidden_factor = float(head_cfg["head_hidden_factor"])

        self.eta_deep   = eta_deep
        self.delta_deep = delta_deep

        # σ_e: always simple linear head from trunk
        self.head_e = nn.Linear(trunk_out_dim, 1)

        # σ_η head: shallow vs deep
        if eta_deep:
            hidden_eta = max(8, int(head_hidden_factor * trunk_out_dim))
            self.head_eta = nn.Sequential(
                nn.Linear(trunk_out_dim, hidden_eta),
                nn.ReLU(),
                nn.Linear(hidden_eta, 1),
            )
        else:
            self.head_eta = nn.Linear(trunk_out_dim, 1)

        # δ head: shallow vs deep
        if delta_deep:
            hidden_d = max(8, int(head_hidden_factor * trunk_out_dim))
            self.head_d = nn.Sequential(
                nn.Linear(trunk_out_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, 1),
            )
        else:
            self.head_d = nn.Linear(trunk_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) → (batch, 3) [z_e, z_eta, z_d] in normalised space.
        """
        h_in = x
        h = self.trunk(x)

        if self.residual and h.shape[1] == h_in.shape[1]:
            h = h + h_in

        e   = self.head_e(h)
        eta = self.head_eta(h)
        d   = self.head_d(h)
        out = torch.cat([e, eta, d], dim=1)
        return out



# TRAINING ONE CONFIG

def train_one_config(
    trunk_cfg: Dict[str, Any],
    head_cfg: Dict[str, Any],
    data: Dict[str, Any],
    batch_size: int = 512,
    max_epochs: int = 80,
    patience: int = 10,
    seed: int = 123,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train one architecture config and return validation metrics
    on ORIGINAL parameter scale.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = data["X_train"].to(DEVICE)
    y_train = data["y_train"].to(DEVICE)
    X_val   = data["X_val"].to(DEVICE)
    y_val   = data["y_val"].to(DEVICE)

    theta_val = data["theta_val"]
    target_scaler = data["target_scaler"]

    n_features = X_train.shape[1]

    model = MLPTrunkHeadModel(
        input_dim=n_features,
        trunk_cfg=trunk_cfg,
        head_cfg=head_cfg,
    ).to(DEVICE)

    lambda_d = float(trunk_cfg["lambda_d"])
    lr       = float(trunk_cfg["lr"])
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
    best_epoch = -1
    best_state = None
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

        # validation
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

        if val_loss_float < best_val_loss - 1e-6:
            best_val_loss = val_loss_float
            best_epoch = epoch
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics on ORIGINAL scale
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).cpu().numpy()

    z_val_hat  = target_scaler.inverse_transform(pred_val)
    z_val_true = target_scaler.inverse_transform(data["y_val"].cpu().numpy())

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
    }

    result.update(trunk_cfg)
    result.update(head_cfg)
    result["best_epoch"] = best_epoch

    return result
