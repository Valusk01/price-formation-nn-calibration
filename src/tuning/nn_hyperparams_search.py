import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DTYPE = np.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model: shared trunk + 3 heads

class MomentsToParamsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, dropout, residual):
        """
        input_dim : # of moments
        hidden_dim: width of hidden layers
        depth     : number of hidden layers in trunk
        dropout   : dropout probability in trunk
        residual  : if True, add a linear residual layer from input → outputs
        """
        super().__init__()
        self.residual = residual

        layers = []
        last_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # 3 heads: σ_e, σ_η, δ
        self.head_e   = nn.Linear(hidden_dim, 1)
        self.head_eta = nn.Linear(hidden_dim, 1)
        self.head_d   = nn.Linear(hidden_dim, 1)

        if residual:
            self.residual_layer = nn.Linear(input_dim, 3)
        else:
            self.residual_layer = None

    def forward(self, x):
        """
        x : (B, input_dim)
        returns: (B, 3)  [z_e, z_eta, z_d]
        """
        h = self.trunk(x)
        out_e   = self.head_e(h)
        out_eta = self.head_eta(h)
        out_d   = self.head_d(h)
        trunk_out = torch.cat([out_e, out_eta, out_d], dim=1)

        if self.residual_layer is not None:
            res = self.residual_layer(x)
            return trunk_out + res
        else:
            return trunk_out



# Scaling helpers (features & targets)

def standardize_features(X_train, X_val):
    """
    Standardise features using TRAIN statistics.

    X_train, X_val : np.ndarray
    returns: X_train_std, X_val_std, mu, std
    """
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0

    X_train_std = (X_train - mu) / std
    X_val_std   = (X_val   - mu) / std
    return X_train_std, X_val_std, mu, std


def transform_targets(theta_train, theta_val):
    """
    Transform parameters to a nicer space and standardise.

    theta_* : (n,3) with columns [sigma_e, sigma_eta, delta]

    For now:
      z_e   = log(sigma_e)
      z_eta = log(sigma_eta)
      z_d   = delta   (no logit yet)

    returns:
      Z_train_std, Z_val_std, mu, std
    """
    sigma_e_train   = theta_train[:, 0]
    sigma_eta_train = theta_train[:, 1]
    delta_train     = theta_train[:, 2]

    sigma_e_val   = theta_val[:, 0]
    sigma_eta_val = theta_val[:, 1]
    delta_val     = theta_val[:, 2]

    # transforms
    z_e_train   = np.log(sigma_e_train)
    z_eta_train = np.log(sigma_eta_train)
    z_d_train   = delta_train

    z_e_val   = np.log(sigma_e_val)
    z_eta_val = np.log(sigma_eta_val)
    z_d_val   = delta_val

    Z_train = np.stack([z_e_train, z_eta_train, z_d_train], axis=1)
    Z_val   = np.stack([z_e_val,   z_eta_val,   z_d_val],   axis=1)

    # standardise per component with TRAIN stats
    mu = Z_train.mean(axis=0, keepdims=True)
    std = Z_train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0

    Z_train_std = (Z_train - mu) / std
    Z_val_std   = (Z_val   - mu) / std

    return Z_train_std, Z_val_std, mu, std


# Train one config

def train_one_config(
    X_train,
    Z_train,
    X_val,
    Z_val,
    depth,
    width,
    dropout,
    residual,
    lr,
    weight_decay,
    lambdas,               # (lambda_e, lambda_eta, lambda_delta)
    n_epochs=80,
    batch_size=512,
    patience=10,
):
    """
    Train a single configuration and return best validation loss and per-parameter MSE.

    All arrays are numpy; training happens inside here with PyTorch.
    """
    n_features = X_train.shape[1]

    model = MomentsToParamsNet(
        input_dim=n_features,
        hidden_dim=width,
        depth=depth,
        dropout=dropout,
        residual=residual,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss(reduction="mean")

    # data loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Z_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Z_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    lambda_e, lambda_eta, lambda_d = lambdas

    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # ---------- train ----------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)  # (B,3)

            optimizer.zero_grad()
            yhat = model(xb)    # (B,3)

            yhat_e   = yhat[:, 0]
            yhat_eta = yhat[:, 1]
            yhat_d   = yhat[:, 2]

            y_e   = yb[:, 0]
            y_eta = yb[:, 1]
            y_d   = yb[:, 2]

            mse_e   = criterion(yhat_e,   y_e)
            mse_eta = criterion(yhat_eta, y_eta)
            mse_d   = criterion(yhat_d,   y_d)

            loss = (
                lambda_e   * mse_e +
                lambda_eta * mse_eta +
                lambda_d   * mse_d
            )
            loss.backward()
            optimizer.step()

        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_mse_e   = []
            val_mse_eta = []
            val_mse_d   = []

            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                yhat = model(xb)

                yhat_e   = yhat[:, 0]
                yhat_eta = yhat[:, 1]
                yhat_d   = yhat[:, 2]

                y_e   = yb[:, 0]
                y_eta = yb[:, 1]
                y_d   = yb[:, 2]

                mse_e   = criterion(yhat_e,   y_e)
                mse_eta = criterion(yhat_eta, y_eta)
                mse_d   = criterion(yhat_d,   y_d)

                total_val_loss = (
                    lambda_e   * mse_e +
                    lambda_eta * mse_eta +
                    lambda_d   * mse_d
                )

                val_losses.append(total_val_loss.item())
                val_mse_e.append(mse_e.item())
                val_mse_eta.append(mse_eta.item())
                val_mse_d.append(mse_d.item())

            mean_val_loss = np.mean(val_losses)
            mean_mse_e    = np.mean(val_mse_e)
            mean_mse_eta  = np.mean(val_mse_eta)
            mean_mse_d    = np.mean(val_mse_d)

        # early stopping
        if mean_val_loss < best_val_loss - 1e-6:
            best_val_loss = mean_val_loss
            best_state = {
                "val_loss": best_val_loss,
                "mse_e": mean_mse_e,
                "mse_eta": mean_mse_eta,
                "mse_d": mean_mse_d,
            }
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # return only metrics (no model state for now)
    result = {
        "best_val_loss": best_state["val_loss"],
        "val_mse_e": best_state["mse_e"],
        "val_mse_eta": best_state["mse_eta"],
        "val_mse_d": best_state["mse_d"],
        "best_epoch": best_epoch,
    }
    return result



# Grid search driver 

def grid_search(
    X,
    theta,
    hyper_grid,
    train_frac=0.7,
    n_epochs=80,
    batch_size=512,
    patience=10,
    random_seed=42,
):
    """
    Run grid search over a list of hyperparameter dicts.

    X      : (n_theta, n_features) moments
    theta  : (n_theta, 3)          parameters
    hyper_grid : list of dicts, each with keys:
        {
          "depth": int,
          "width": int,
          "dropout": float,
          "residual": bool,
          "lr": float,
          "weight_decay": float,
          "lambda_e": float,
          "lambda_eta": float,
          "lambda_d": float
        }

    Returns: list of result dicts (one per config).
    """
    n_total = X.shape[0]
    n_train = int(train_frac * n_total)
    n_val   = n_total - n_train

    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    idx_train = indices[:n_train]
    idx_val   = indices[n_train:]

    X_train_raw = X[idx_train]
    X_val_raw   = X[idx_val]
    theta_train = theta[idx_train]
    theta_val   = theta[idx_val]

    # feature scaling
    X_train, X_val, feat_mu, feat_std = standardize_features(X_train_raw, X_val_raw)

    # target transform + scaling
    Z_train, Z_val, target_mu, target_std = transform_targets(theta_train, theta_val)

    results = []
    total_configs = len(hyper_grid)

    for idx, cfg in enumerate(hyper_grid, start=1):
        depth        = cfg["depth"]
        width        = cfg["width"]
        dropout      = cfg["dropout"]
        residual     = cfg["residual"]
        lr           = cfg["lr"]
        weight_decay = cfg["weight_decay"]
        lambda_e     = cfg["lambda_e"]
        lambda_eta   = cfg["lambda_eta"]
        lambda_d     = cfg["lambda_d"]

        print(
            f"\nConfig {idx}/{total_configs}: "
            f"depth={depth}, width={width}, dropout={dropout}, "
            f"residual={residual}, lr={lr}, wd={weight_decay}, "
            f"lmb=({lambda_e},{lambda_eta},{lambda_d})"
        )

        res = train_one_config(
            X_train=X_train,
            Z_train=Z_train,
            X_val=X_val,
            Z_val=Z_val,
            depth=depth,
            width=width,
            dropout=dropout,
            residual=residual,
            lr=lr,
            weight_decay=weight_decay,
            lambdas=(lambda_e, lambda_eta, lambda_d),
            n_epochs=n_epochs,
            batch_size=batch_size,
            patience=patience,
        )

        out = dict(cfg)  # copy hyperparameters
        out.update(res)  # add metrics
        results.append(out)

    meta = {
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "train_frac": float(train_frac),
        "random_seed": int(random_seed),
        "feat_mu": feat_mu,
        "feat_std": feat_std,
        "target_mu": target_mu,
        "target_std": target_std,
    }

    return results, meta
