# CNN_train.py
"""Two-stage HPGe waveform localization with learned coordinate uncertainty.

Stage 1:
    Train backbone + coordinate head with normalized-coordinate MSE.
Stage 2:
    Reload best Stage-1 checkpoint, freeze backbone and coordinate head,
    and train only an uncertainty head with Gaussian NLL.

Input:  [batch, 9 contacts, 100 samples]
Output: x, y, z in mm and sigma_x, sigma_y, sigma_z in mm
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    data_dir: str = "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset/pytorch_export_v2"
    output_dir: str = "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset/cnn_two_stage"

    batch_size: int = 256
    num_workers: int = 4
    seed: int = 74037
    dropout: float = 0.15
    pool_bins: int = 15
    standardize_waveforms: bool = True

    stage1_epochs: int = 60
    stage1_learning_rate: float = 1e-3
    stage1_weight_decay: float = 1e-4
    stage1_scheduler_patience: int = 4
    stage1_early_stopping_patience: int = 30

    stage2_epochs: int = 40
    stage2_learning_rate: float = 5e-4
    stage2_weight_decay: float = 1e-4
    stage2_scheduler_patience: int = 5
    stage2_early_stopping_patience: int = 25

    minimum_log_variance: float = -10.0
    maximum_log_variance: float = 5.0


# ============================ Data ============================

def parse_metadata(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def load_vector(path: Path, dtype, expected_length=None):
    x = np.fromfile(path, dtype=dtype)
    if expected_length is not None and x.size != expected_length:
        raise ValueError(f"{path}: found {x.size}, expected {expected_length}")
    return x


class WaveformDataset(Dataset):
    def __init__(self, X, y_mm, indices, channel_mean, channel_std,
                 position_mean, position_std, standardize_waveforms):
        self.X = X
        self.y_mm = y_mm
        self.indices = np.asarray(indices, dtype=np.int64)
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32).reshape(9, 1)
        self.channel_std = np.maximum(
            np.asarray(channel_std, dtype=np.float32).reshape(9, 1), 1e-6
        )
        self.position_mean = np.asarray(position_mean, dtype=np.float32)
        self.position_std = np.maximum(np.asarray(position_std, dtype=np.float32), 1e-6)
        self.standardize_waveforms = standardize_waveforms

    def __len__(self):
        return self.indices.size

    def __getitem__(self, item):
        index = int(self.indices[item])
        waveform = np.array(self.X[index], dtype=np.float32, copy=True)
        if self.standardize_waveforms:
            waveform = (waveform - self.channel_mean) / self.channel_std
        position_mm = np.array(self.y_mm[index], dtype=np.float32, copy=True)
        position_normalized = (position_mm - self.position_mean) / self.position_std
        return torch.from_numpy(waveform), torch.from_numpy(position_normalized), index


# ============================ Model ============================

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class WaveformBackbone(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
        self.pool_bins = pool_bins
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            ResidualBlock1D(64, 5, dropout),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            ResidualBlock1D(128, 5, dropout),
            nn.Conv1d(128, 192, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            ResidualBlock1D(192, 3, dropout),
            nn.AdaptiveAvgPool1d(pool_bins),
            nn.Flatten(),
        )
        self.output_features = 192 * pool_bins

    def forward(self, x):
        return self.net(x)


class CoordinateHead(nn.Module):
    def __init__(self, input_features: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, features):
        return self.net(features)


class UncertaintyHead(nn.Module):
    def __init__(self, input_features: int, dropout: float):
        super().__init__()
        # The fixed coordinate estimate is appended to the frozen CNN features.
        self.net = nn.Sequential(
            nn.Linear(input_features + 3, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, features, coordinate_mean):
        return self.net(torch.cat([features, coordinate_mean], dim=1))


class TwoStageModel(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
        self.backbone = WaveformBackbone(dropout, pool_bins)
        self.coordinate_head = CoordinateHead(self.backbone.output_features, dropout)
        self.uncertainty_head = UncertaintyHead(self.backbone.output_features, dropout)

    def coordinate_forward(self, x):
        features = self.backbone(x)
        coordinate_mean = self.coordinate_head(features)
        return coordinate_mean, features

    def uncertainty_forward(self, x):
        # Stage 2 must not update the coordinate solution or backbone statistics.
        with torch.no_grad():
            features = self.backbone(x)
            coordinate_mean = self.coordinate_head(features)
        log_variance = self.uncertainty_head(features.detach(), coordinate_mean.detach())
        return coordinate_mean.detach(), log_variance

    def forward(self, x):
        features = self.backbone(x)
        coordinate_mean = self.coordinate_head(features)
        log_variance = self.uncertainty_head(features, coordinate_mean)
        return coordinate_mean, log_variance


class FixedMeanGaussianNLL(nn.Module):
    def __init__(self, minimum_log_variance: float, maximum_log_variance: float):
        super().__init__()
        self.minimum_log_variance = minimum_log_variance
        self.maximum_log_variance = maximum_log_variance

    def forward(self, coordinate_mean, log_variance, target):
        log_variance = torch.clamp(
            log_variance, self.minimum_log_variance, self.maximum_log_variance
        )
        squared_error = (target - coordinate_mean) ** 2
        return 0.5 * (
            torch.exp(-log_variance) * squared_error + log_variance
        ).mean()


# ============================ Utilities ============================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(x, mean, std):
    return x * std.reshape(1, 3) + mean.reshape(1, 3)


def coordinate_metrics(pred_mm, truth_mm):
    residual = pred_mm - truth_mm
    distance = np.linalg.norm(residual, axis=1)
    return {
        "mae_x_mm": float(np.mean(np.abs(residual[:, 0]))),
        "mae_y_mm": float(np.mean(np.abs(residual[:, 1]))),
        "mae_z_mm": float(np.mean(np.abs(residual[:, 2]))),
        "rmse_x_mm": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "rmse_y_mm": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "rmse_z_mm": float(np.sqrt(np.mean(residual[:, 2] ** 2))),
        "mean_3d_error_mm": float(np.mean(distance)),
        "median_3d_error_mm": float(np.median(distance)),
        "p68_3d_error_mm": float(np.percentile(distance, 68)),
        "p90_3d_error_mm": float(np.percentile(distance, 90)),
    }


@torch.no_grad()
def evaluate_coordinates(model, loader, device, position_mean, position_std):
    model.eval()
    mse_total = 0.0
    count = 0
    pred_all, truth_all, index_all = [], [], []
    loss_fn = nn.MSELoss()

    for X, target, indices in loader:
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred, _ = model.coordinate_forward(X)
        loss = loss_fn(pred, target)
        batch = X.shape[0]
        mse_total += float(loss.item()) * batch
        count += batch
        pred_all.append(pred.cpu().numpy())
        truth_all.append(target.cpu().numpy())
        index_all.append(np.asarray(indices))

    pred_norm = np.concatenate(pred_all)
    truth_norm = np.concatenate(truth_all)
    indices = np.concatenate(index_all).astype(np.int64)
    pred_mm = denormalize(pred_norm, position_mean, position_std)
    truth_mm = denormalize(truth_norm, position_mean, position_std)
    metrics = {"coordinate_mse_normalized": mse_total / count}
    metrics.update(coordinate_metrics(pred_mm, truth_mm))
    return metrics, pred_mm, truth_mm, indices


@torch.no_grad()
def evaluate_uncertainty(model, loader, loss_fn, device, position_mean,
                         position_std, min_log_variance, max_log_variance):
    model.eval()
    nll_total = 0.0
    count = 0
    pred_all, truth_all, logvar_all, index_all = [], [], [], []

    for X, target, indices in loader:
        X = X.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred, log_variance = model.uncertainty_forward(X)
        loss = loss_fn(pred, log_variance, target)
        batch = X.shape[0]
        nll_total += float(loss.item()) * batch
        count += batch
        pred_all.append(pred.cpu().numpy())
        truth_all.append(target.cpu().numpy())
        logvar_all.append(log_variance.cpu().numpy())
        index_all.append(np.asarray(indices))

    pred_norm = np.concatenate(pred_all)
    truth_norm = np.concatenate(truth_all)
    logvar_norm = np.clip(
        np.concatenate(logvar_all), min_log_variance, max_log_variance
    )
    indices = np.concatenate(index_all).astype(np.int64)
    pred_mm = denormalize(pred_norm, position_mean, position_std)
    truth_mm = denormalize(truth_norm, position_mean, position_std)
    sigma_mm = np.exp(0.5 * logvar_norm) * position_std.reshape(1, 3)

    residual = pred_mm - truth_mm
    abs_residual = np.abs(residual)
    standardized = residual / np.maximum(sigma_mm, 1e-6)
    within1 = abs_residual <= sigma_mm
    within2 = abs_residual <= 2.0 * sigma_mm

    metrics = {"gaussian_nll_normalized": nll_total / count}
    metrics.update(coordinate_metrics(pred_mm, truth_mm))
    for axis, name in enumerate(("x", "y", "z")):
        metrics[f"median_sigma_{name}_mm"] = float(np.median(sigma_mm[:, axis]))
        metrics[f"coverage_1sigma_{name}"] = float(np.mean(within1[:, axis]))
        metrics[f"coverage_2sigma_{name}"] = float(np.mean(within2[:, axis]))
        metrics[f"standardized_residual_rms_{name}"] = float(
            np.sqrt(np.mean(standardized[:, axis] ** 2))
        )
        metrics[f"uncertainty_abs_error_correlation_{name}"] = float(
            np.corrcoef(sigma_mm[:, axis], abs_residual[:, axis])[0, 1]
        )
    metrics["coverage_1sigma_all_coordinates"] = float(np.mean(np.all(within1, axis=1)))
    metrics["coverage_2sigma_all_coordinates"] = float(np.mean(np.all(within2, axis=1)))
    metrics["uncertainty_3d_error_correlation"] = float(
        np.corrcoef(
            np.sqrt(np.sum(sigma_mm ** 2, axis=1)),
            np.linalg.norm(residual, axis=1),
        )[0, 1]
    )
    return metrics, pred_mm, truth_mm, sigma_mm, indices


def set_stage2_frozen(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.coordinate_head.parameters():
        parameter.requires_grad = False
    for parameter in model.uncertainty_head.parameters():
        parameter.requires_grad = True

    # Frozen BatchNorm running statistics must remain fixed during stage 2.
    model.backbone.eval()
    model.coordinate_head.eval()
    model.uncertainty_head.train()


def save_predictions(path, indices, truth_mm, pred_mm, sigma_mm):
    residual = pred_mm - truth_mm
    distance = np.linalg.norm(residual, axis=1)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_index", "true_x_mm", "true_y_mm", "true_z_mm",
            "pred_x_mm", "pred_y_mm", "pred_z_mm",
            "sigma_x_mm", "sigma_y_mm", "sigma_z_mm",
            "error_x_mm", "error_y_mm", "error_z_mm", "error_3d_mm",
        ])
        for i in range(len(indices)):
            writer.writerow([
                int(indices[i]), *truth_mm[i], *pred_mm[i], *sigma_mm[i],
                *residual[i], float(distance[i]),
            ])


# ============================ Training ============================

def main():
    config = Config()
    seed_everything(config.seed)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = parse_metadata(data_dir / "metadata.txt")
    n = int(metadata["n_events"])
    contacts = int(metadata["n_contacts"])
    samples = int(metadata["n_samples"])
    if (contacts, samples) != (9, 100):
        raise ValueError(f"Expected waveform shape (9,100), found {(contacts,samples)}")

    X = np.memmap(
        data_dir / "waveforms_f32.bin", dtype=np.float32, mode="r",
        shape=(n, contacts, samples), order="C"
    )
    y_mm = np.memmap(
        data_dir / "positions_f32.bin", dtype=np.float32, mode="r",
        shape=(n, 3), order="C"
    )
    channel_mean = load_vector(data_dir / "channel_mean_f32.bin", np.float32, 9)
    channel_std = load_vector(data_dir / "channel_std_f32.bin", np.float32, 9)
    train_idx = load_vector(data_dir / "train_indices_i64.bin", np.int64)
    val_idx = load_vector(data_dir / "validation_indices_i64.bin", np.int64)
    test_idx = load_vector(data_dir / "test_indices_i64.bin", np.int64)

    position_mean = np.mean(np.asarray(y_mm[train_idx]), axis=0).astype(np.float32)
    position_std = np.std(np.asarray(y_mm[train_idx]), axis=0).astype(np.float32)
    position_std = np.maximum(position_std, 1e-6)

    train_set = WaveformDataset(
        X, y_mm, train_idx, channel_mean, channel_std,
        position_mean, position_std, config.standardize_waveforms
    )
    val_set = WaveformDataset(
        X, y_mm, val_idx, channel_mean, channel_std,
        position_mean, position_std, config.standardize_waveforms
    )
    test_set = WaveformDataset(
        X, y_mm, test_idx, channel_mean, channel_std,
        position_mean, position_std, config.standardize_waveforms
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStageModel(config.dropout, config.pool_bins).to(device)

    print(f"Device: {device}")
    print(f"Events: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")
    print(f"Pool bins: {config.pool_bins}")
    print("Stage 1: train backbone + coordinate head with MSE")

    # ---------- Stage 1 ----------
    stage1_parameters = list(model.backbone.parameters()) + list(model.coordinate_head.parameters())
    optimizer1 = torch.optim.AdamW(
        stage1_parameters,
        lr=config.stage1_learning_rate,
        weight_decay=config.stage1_weight_decay,
    )
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="min", factor=0.5,
        patience=config.stage1_scheduler_patience, min_lr=1e-6
    )
    mse_loss = nn.MSELoss()
    best_stage1 = float("inf")
    bad_stage1 = 0
    stage1_path = output_dir / "best_stage1_coordinate_model.pt"
    stage1_history = []

    for epoch in range(1, config.stage1_epochs + 1):
        model.backbone.train()
        model.coordinate_head.train()
        model.uncertainty_head.eval()
        total = 0.0
        count = 0

        for X_batch, target, _ in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer1.zero_grad(set_to_none=True)
            pred, _ = model.coordinate_forward(X_batch)
            loss = mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1_parameters, 5.0)
            optimizer1.step()
            total += float(loss.item()) * X_batch.shape[0]
            count += X_batch.shape[0]

        train_mse = total / count
        val_metrics, _, _, _ = evaluate_coordinates(
            model, val_loader, device, position_mean, position_std
        )
        val_mse = val_metrics["coordinate_mse_normalized"]
        scheduler1.step(val_mse)
        lr = optimizer1.param_groups[0]["lr"]
        stage1_history.append({
            "epoch": epoch, "train_mse_normalized": train_mse,
            "learning_rate": lr, **val_metrics,
        })
        print(
            f"Stage1 {epoch:03d} | trainMSE={train_mse:.6f} | "
            f"valMSE={val_mse:.6f} | "
            f"median3D={val_metrics['median_3d_error_mm']:.3f} mm | lr={lr:.2e}"
        )

        if val_mse < best_stage1 - 1e-7:
            best_stage1 = val_mse
            bad_stage1 = 0
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": model.backbone.state_dict(),
                "coordinate_head_state_dict": model.coordinate_head.state_dict(),
                "validation_metrics": val_metrics,
                "position_mean_mm": position_mean,
                "position_std_mm": position_std,
                "channel_mean_keV": channel_mean,
                "channel_std_keV": channel_std,
                "config": asdict(config),
            }, stage1_path)
        else:
            bad_stage1 += 1
            if bad_stage1 >= config.stage1_early_stopping_patience:
                print(f"Stage 1 early stopping after epoch {epoch}")
                break

    # Restore best coordinate solution before training uncertainty.
    stage1_checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
    model.backbone.load_state_dict(stage1_checkpoint["backbone_state_dict"])
    model.coordinate_head.load_state_dict(stage1_checkpoint["coordinate_head_state_dict"])

    stage1_test_metrics, _, _, _ = evaluate_coordinates(
        model, test_loader, device, position_mean, position_std
    )
    print("\nStage 1 coordinate test metrics:")
    for key, value in stage1_test_metrics.items():
        print(f"  {key}: {value:.6f}")

    # ---------- Stage 2 ----------
    print("\nStage 2: freeze coordinates and train uncertainty head with Gaussian NLL")
    set_stage2_frozen(model)
    optimizer2 = torch.optim.AdamW(
        model.uncertainty_head.parameters(),
        lr=config.stage2_learning_rate,
        weight_decay=config.stage2_weight_decay,
    )
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.5,
        patience=config.stage2_scheduler_patience, min_lr=1e-6
    )
    nll_loss = FixedMeanGaussianNLL(
        config.minimum_log_variance, config.maximum_log_variance
    )
    best_stage2 = float("inf")
    bad_stage2 = 0
    stage2_path = output_dir / "best_stage2_uncertainty_model.pt"
    stage2_history = []

    for epoch in range(1, config.stage2_epochs + 1):
        # Calling model.train() would alter frozen BatchNorm statistics, so set
        # each component explicitly on every epoch.
        set_stage2_frozen(model)
        total = 0.0
        count = 0

        for X_batch, target, _ in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer2.zero_grad(set_to_none=True)
            coordinate_mean, log_variance = model.uncertainty_forward(X_batch)
            loss = nll_loss(coordinate_mean, log_variance, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.uncertainty_head.parameters(), 5.0)
            optimizer2.step()
            total += float(loss.item()) * X_batch.shape[0]
            count += X_batch.shape[0]

        train_nll = total / count
        val_metrics, _, _, _, _ = evaluate_uncertainty(
            model, val_loader, nll_loss, device, position_mean, position_std,
            config.minimum_log_variance, config.maximum_log_variance
        )
        val_nll = val_metrics["gaussian_nll_normalized"]
        scheduler2.step(val_nll)
        lr = optimizer2.param_groups[0]["lr"]
        stage2_history.append({
            "epoch": epoch, "train_nll_normalized": train_nll,
            "learning_rate": lr, **val_metrics,
        })
        print(
            f"Stage2 {epoch:03d} | trainNLL={train_nll:.6f} | "
            f"valNLL={val_nll:.6f} | "
            f"medianSigma=({val_metrics['median_sigma_x_mm']:.3f}, "
            f"{val_metrics['median_sigma_y_mm']:.3f}, "
            f"{val_metrics['median_sigma_z_mm']:.3f}) mm | lr={lr:.2e}"
        )

        if val_nll < best_stage2 - 1e-7:
            best_stage2 = val_nll
            bad_stage2 = 0
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": model.backbone.state_dict(),
                "coordinate_head_state_dict": model.coordinate_head.state_dict(),
                "uncertainty_head_state_dict": model.uncertainty_head.state_dict(),
                "validation_metrics": val_metrics,
                "position_mean_mm": position_mean,
                "position_std_mm": position_std,
                "channel_mean_keV": channel_mean,
                "channel_std_keV": channel_std,
                "config": asdict(config),
            }, stage2_path)
        else:
            bad_stage2 += 1
            if bad_stage2 >= config.stage2_early_stopping_patience:
                print(f"Stage 2 early stopping after epoch {epoch}")
                break

    # Final test uses best Stage-2 uncertainty checkpoint and unchanged Stage-1 coordinates.
    checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    model.coordinate_head.load_state_dict(checkpoint["coordinate_head_state_dict"])
    model.uncertainty_head.load_state_dict(checkpoint["uncertainty_head_state_dict"])

    test_metrics, pred_mm, truth_mm, sigma_mm, indices = evaluate_uncertainty(
        model, test_loader, nll_loss, device, position_mean, position_std,
        config.minimum_log_variance, config.maximum_log_variance
    )

    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output_dir / "stage1_history.json").write_text(json.dumps(stage1_history, indent=2))
    (output_dir / "stage2_history.json").write_text(json.dumps(stage2_history, indent=2))
    (output_dir / "stage1_test_metrics.json").write_text(
        json.dumps(stage1_test_metrics, indent=2)
    )
    (output_dir / "two_stage_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )
    save_predictions(
        output_dir / "test_predictions_with_uncertainty.csv",
        indices, truth_mm, pred_mm, sigma_mm
    )

    print("\nFinal two-stage test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nStage 1 checkpoint: {stage1_path}")
    print(f"Stage 2 checkpoint: {stage2_path}")


if __name__ == "__main__":
    main()