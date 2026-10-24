"""Per-band concentration regressors.

One small MLP per band maps raw ``[Control, Signal]`` intensities to a
concentration. Each model is trained with K-fold CV for evaluation, then a
final model is refit on all data and saved together with its scalers.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .visualization import plot_regression_diagnostics


class Net(nn.Module):
    """2 -> 64 -> 32 -> 16 -> 1 MLP."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def _fit(model, loader, epochs, lr, huber_delta):
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def train_regressor_cv(df, target_col="Target", control_col="Control", signal_col="Signal",
                       n_splits=5, epochs=150, batch_size=32, lr=0.001, huber_delta=1.0,
                       seed=42, exclude=None, plot_dir=None, plot_prefix="", show_plots=False,
                       verbose=True):
    """K-fold CV, then refit on all data.

    Returns a dict with ``best_fold``, ``fold_results`` (array of MAE/RMSE/R2),
    ``best_model``, ``final_model``, ``scaler_X``, ``scaler_y``, ``pred``, ``true``.
    """
    df_clean = df[~df[target_col].isin(exclude or [])].copy()

    X = df_clean[[control_col, signal_col]].values
    y = df_clean[target_col].values.reshape(-1, 1)

    if seed is not None:
        torch.manual_seed(seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results, fold_models, fold_scalers = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"\n===== FOLD {fold + 1} =====")

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train = torch.FloatTensor(scaler_X.fit_transform(X[train_idx]))
        y_train = torch.FloatTensor(scaler_y.fit_transform(y[train_idx]))
        X_val = torch.FloatTensor(scaler_X.transform(X[val_idx]))

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        model = Net()
        _fit(model, loader, epochs, lr, huber_delta)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_val).numpy()

        pred = scaler_y.inverse_transform(pred_scaled).flatten()
        true = y[val_idx].flatten()

        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        if verbose:
            print(f"MAE : {mae:.4f}\nRMSE: {rmse:.4f}\nR²  : {r2:.4f}")

        fold_results.append((mae, rmse, r2))
        fold_models.append(model)
        fold_scalers.append((scaler_X, scaler_y))

    best_idx = int(np.argmin([r[0] for r in fold_results]))
    best_model = fold_models[best_idx]
    if verbose:
        print(f"\nBest fold: {best_idx + 1}")

    # Final model: refit on all data
    scaler_X_full, scaler_y_full = StandardScaler(), StandardScaler()
    X_full_t = torch.FloatTensor(scaler_X_full.fit_transform(X))
    y_full_t = torch.FloatTensor(scaler_y_full.fit_transform(y))
    full_loader = DataLoader(TensorDataset(X_full_t, y_full_t), batch_size=batch_size, shuffle=True)

    final_model = Net()
    _fit(final_model, full_loader, epochs, lr, huber_delta)

    final_model.eval()
    with torch.no_grad():
        pred_scaled = final_model(X_full_t).numpy()
    pred = scaler_y_full.inverse_transform(pred_scaled).flatten()
    true = y.flatten()

    if plot_dir is not None or show_plots:
        ratio = df_clean[signal_col] / df_clean[control_col]
        plot_regression_diagnostics(df_clean[target_col], ratio, true, pred,
                                    out_dir=plot_dir, prefix=plot_prefix, show=show_plots)

    fold_results = np.array(fold_results)
    if verbose:
        print("\n==============================")
        print("        FINAL CV RESULTS       ")
        print("==============================")
        for i, name in enumerate(["MAE ", "RMSE", "R²  "]):
            print(f"{name}: {fold_results[:, i].mean():.4f} ± {fold_results[:, i].std():.4f}")
        print("==============================\n")

    return {
        "best_fold": best_idx + 1,
        "fold_results": fold_results,
        "best_model": best_model,
        "final_model": final_model,
        "scaler_X": scaler_X_full,
        "scaler_y": scaler_y_full,
        "pred": pred,
        "true": true,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def band_model_path(models_dir, band):
    return Path(models_dir) / f"final_model_band{band}.pth"


def save_checkpoint(path, results):
    """Save the final model + scaler statistics from ``train_regressor_cv``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": results["final_model"].state_dict(),
        "scaler_X_mean": results["scaler_X"].mean_.tolist(),
        "scaler_X_scale": results["scaler_X"].scale_.tolist(),
        "scaler_y_mean": results["scaler_y"].mean_.tolist(),
        "scaler_y_scale": results["scaler_y"].scale_.tolist(),
    }, str(path))


def load_band_model(model_path, device="cpu"):
    """Load a saved band regressor and its scalers (validated)."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)

    net = Net().to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    for key in ("scaler_X_mean", "scaler_X_scale", "scaler_y_mean", "scaler_y_scale"):
        if key not in checkpoint:
            raise KeyError(f"{model_path.name} missing {key}")

    X_mean = np.asarray(checkpoint["scaler_X_mean"], dtype=np.float32).reshape(1, -1)
    X_scale = np.asarray(checkpoint["scaler_X_scale"], dtype=np.float32).reshape(1, -1)
    y_mean = np.asarray(checkpoint["scaler_y_mean"], dtype=np.float32).reshape(-1)
    y_scale = np.asarray(checkpoint["scaler_y_scale"], dtype=np.float32).reshape(-1)

    if X_mean.shape[1] != 2:
        raise ValueError(f"{model_path.name}: expected 2 input features, got {X_mean.shape[1]}")
    if np.any(X_scale == 0):
        raise ValueError(f"Invalid input scaler in {model_path.name}")
    if np.any(y_scale == 0):
        raise ValueError(f"Invalid output scaler in {model_path.name}")

    return {"model": net, "X_mean": X_mean, "X_scale": X_scale,
            "y_mean": y_mean, "y_scale": y_scale}


def predict_band(control_intensity, signal_intensity, model_info):
    """Predict a concentration from raw ``[Control, Signal]`` (training units/order)."""
    X = np.array([[control_intensity, signal_intensity]], dtype=np.float32)
    X_scaled = (X - model_info["X_mean"]) / model_info["X_scale"]

    with torch.no_grad():
        y_scaled = model_info["model"](torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()

    prediction = y_scaled * model_info["y_scale"] + model_info["y_mean"]
    return float(prediction.reshape(-1)[0])
