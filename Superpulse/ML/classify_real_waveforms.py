# classify_real_waveforms.py
"""Classify real 9-contact HPGe waveforms with the trained two-stage CNN.

Outputs:
  real_waveform_predictions.csv
  real_hits_xy_uncertainty.png
  real_hits_xy_uncertainty_clipped.png
  real_uncertainty_histogram.png

The inferred coordinates are x_mm, y_mm, z_ssd_mm. Uncertainty colors use
sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2), which is a scalar uncertainty proxy.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ============================ User settings ============================

REAL_WAVEFORM_FILE = Path(
    "/data1/flerner/hpge_sims/CNN/realdata/out50k.dat"
)

MODEL_CHECKPOINT = Path(
    "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset/"
    "cnn_two_stage/best_stage2_uncertainty_model.pt"
)

OUTPUT_DIR = Path(
    "/data1/flerner/hpge_sims/CNN/real_waveform_predictions_two_stage"
)

# Must match training architecture.
POOL_BINS = 15
DROPOUT = 0.15
NCONTACTS = 9
NSAMPLES = 100
CORE_CONTACT_ID = 9
OUTER_CONTACT_IDS = tuple(range(1, 9))
ALL_CONTACT_IDS = tuple(range(1, 10))

# Real-waveform preprocessing, using 1-based sample definitions from Julia.
BASELINE_START_SAMPLE = 1
BASELINE_END_SAMPLE = 30
FINAL_START_SAMPLE = 90
FINAL_END_SAMPLE = 99
SAMPLE_PERIOD_NS = 10.0

# Real ADC-to-keV calibration used by the superpulse builder.
PHOTOPEAK_ENERGY_KEV = 1332.5
WAVEFORM_1332_ADC = {
    1: 980.3978243978245,
    2: 1329.093149540518,
    3: 986.625641025641,
    4: 972.1213696369637,
    5: 995.7892176199867,
    6: 974.9032280701754,
    7: 978.573590504451,
    8: 990.8224089635855,
    9: 1012.8978067318131,
}
CONTACT_ADC_TO_KEV = {
    cid: PHOTOPEAK_ENERGY_KEV / WAVEFORM_1332_ADC[cid]
    for cid in ALL_CONTACT_IDS
}

# Optional event selection. Set to False to classify every complete event.
APPLY_MIN_ENERGY_CUT = True
MIN_EVENT_ENERGY_KEV = 20.0
APPLY_MAX_ENERGY_CUT = False
MAX_EVENT_ENERGY_KEV = 3000.0
APPLY_FINAL_CONFIDENCE_CUT = False
MIN_FINAL_CONFIDENCE = 0.90
MAX_EVENTS = 100000  # Example: 10000, or None for all events.

# Real data should already have the same core polarity as the fitted real data.
FLIP_REAL_CORE_POLARITY = False

# Optional event-level timing alignment. Enable only if raw events are not
# already aligned by acquisition. Positive shift moves a waveform later.
ALIGN_REAL_EVENTS = False
ALIGNMENT_FRACTION = 0.30
ALIGNMENT_REFERENCE_SAMPLE = 50.0
TIMING_SMOOTH_HALF_WINDOW = 1
MAX_ABSOLUTE_ALIGNMENT_SHIFT_SAMPLES = 4.0

# If a validation-derived uncertainty calibration scale is not stored in the
# checkpoint, use identity scaling. Replace only with validation-set factors.
FALLBACK_UNCERTAINTY_CALIBRATION_SCALE = np.array(
    [1.0, 1.0, 1.0], dtype=np.float32
)

# Plot choices.
UNCERTAINTY_CLIP_PERCENTILE = 95.0
MARKER_SIZE = 8.0
MARKER_ALPHA = 0.75


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class WaveformBackbone(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class UncertaintyHead(nn.Module):
    def __init__(self, input_features: int, dropout: float):
        super().__init__()
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

    def forward(
        self, features: torch.Tensor, coordinate_mean: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([features, coordinate_mean], dim=1))


class TwoStageModel(nn.Module):
    def __init__(self, dropout: float, pool_bins: int):
        super().__init__()
        self.backbone = WaveformBackbone(dropout, pool_bins)
        self.coordinate_head = CoordinateHead(
            self.backbone.output_features, dropout
        )
        self.uncertainty_head = UncertaintyHead(
            self.backbone.output_features, dropout
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        coordinate_mean = self.coordinate_head(features)
        log_variance = self.uncertainty_head(features, coordinate_mean)
        return coordinate_mean, log_variance


# ============================ Parsing ============================

LINE_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*,\s*"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"\s*:\s*(.*)$"
)


def label_to_contact_id(label: str) -> int:
    return CORE_CONTACT_ID if label.strip().upper() == "FV" else int(label)


def fix_waveform_length(values: np.ndarray) -> np.ndarray:
    if values.size == NSAMPLES:
        return values.astype(np.float32, copy=False)
    if values.size > NSAMPLES:
        return values[:NSAMPLES].astype(np.float32, copy=False)
    if values.size == 0:
        return np.zeros(NSAMPLES, dtype=np.float32)
    return np.pad(
        values.astype(np.float32, copy=False),
        (0, NSAMPLES - values.size),
        mode="edge",
    )


def read_real_events(path: Path) -> List[np.ndarray]:
    events: List[np.ndarray] = []
    current: Dict[int, np.ndarray] = {}

    def flush() -> None:
        nonlocal current
        if not current:
            return
        if all(cid in current for cid in ALL_CONTACT_IDS):
            matrix = np.stack([current[cid] for cid in ALL_CONTACT_IDS], axis=0)
            events.append(matrix.astype(np.float32, copy=False))
        current = {}

    with path.open("r") as handle:
        for line in handle:
            if not line.strip():
                flush()
                continue

            match = LINE_PATTERN.match(line.strip())
            if match is None:
                continue

            label, payload = match.groups()
            try:
                contact_id = label_to_contact_id(label)
            except ValueError:
                continue
            if contact_id not in ALL_CONTACT_IDS:
                continue

            values = np.array(
                [float(value.strip()) for value in payload.split(",") if value.strip()],
                dtype=np.float32,
            )
            if contact_id in current:
                flush()
            current[contact_id] = fix_waveform_length(values)

            if len(current) == NCONTACTS:
                flush()

            if MAX_EVENTS is not None and len(events) >= MAX_EVENTS:
                break

    flush()
    return events[:MAX_EVENTS] if MAX_EVENTS is not None else events


# ============================ Preprocessing ============================

def baseline_subtract_and_calibrate(waveform: np.ndarray) -> np.ndarray:
    output = waveform.astype(np.float32, copy=True)
    baseline_slice = slice(BASELINE_START_SAMPLE - 1, BASELINE_END_SAMPLE)

    for contact_id in ALL_CONTACT_IDS:
        index = contact_id - 1
        baseline = float(np.mean(output[index, baseline_slice]))
        output[index] -= baseline
        output[index] *= np.float32(CONTACT_ADC_TO_KEV[contact_id])

    if FLIP_REAL_CORE_POLARITY:
        output[CORE_CONTACT_ID - 1] *= -1.0

    return output


def centered_moving_average(y: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0:
        return y.astype(np.float32, copy=True)
    result = np.empty_like(y, dtype=np.float32)
    for index in range(y.size):
        low = max(0, index - half_window)
        high = min(y.size, index + half_window + 1)
        result[index] = np.mean(y[low:high])
    return result


def fractional_crossing_sample(y: np.ndarray, fraction: float) -> float:
    baseline = float(np.mean(y[BASELINE_START_SAMPLE - 1 : BASELINE_END_SAMPLE]))
    final = float(np.mean(y[FINAL_START_SAMPLE - 1 : FINAL_END_SAMPLE]))
    amplitude = final - baseline
    if abs(amplitude) < 1e-12:
        return float("nan")
    threshold = baseline + fraction * amplitude

    for index in range(1, y.size):
        crossed = y[index] >= threshold if amplitude > 0 else y[index] <= threshold
        if crossed:
            delta = float(y[index] - y[index - 1])
            if abs(delta) < 1e-12:
                return float(index + 1)
            interpolation = (threshold - float(y[index - 1])) / delta
            # Return Julia-style 1-based fractional sample for consistency.
            return float(index + interpolation)
    return float("nan")


def shift_trace(y: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_axis = np.arange(y.size, dtype=np.float32)
    query_axis = sample_axis - np.float32(shift_samples)
    return np.interp(
        query_axis,
        sample_axis,
        y,
        left=float(y[0]),
        right=float(y[-1]),
    ).astype(np.float32)


def align_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, float, float]:
    if not ALIGN_REAL_EVENTS:
        return waveform, float("nan"), 0.0

    core = centered_moving_average(
        waveform[CORE_CONTACT_ID - 1], TIMING_SMOOTH_HALF_WINDOW
    )
    crossing = fractional_crossing_sample(core, ALIGNMENT_FRACTION)
    if not np.isfinite(crossing):
        return waveform, crossing, 0.0

    shift = ALIGNMENT_REFERENCE_SAMPLE - crossing
    if abs(shift) > MAX_ABSOLUTE_ALIGNMENT_SHIFT_SAMPLES:
        return waveform, crossing, float("nan")

    aligned = np.stack(
        [shift_trace(waveform[channel], shift) for channel in range(NCONTACTS)],
        axis=0,
    )
    return aligned, crossing, float(shift)


def final_levels(waveform: np.ndarray) -> np.ndarray:
    return np.mean(
        waveform[:, FINAL_START_SAMPLE - 1 : FINAL_END_SAMPLE], axis=1
    )


def event_selection(waveform: np.ndarray) -> Tuple[bool, int, float, float]:
    levels = final_levels(waveform)
    target_index = int(np.argmax(np.abs(levels[:8])))
    target_contact = target_index + 1
    energy = abs(float(levels[target_index]))

    other = np.delete(np.abs(levels[:8]), target_index)
    second = float(np.max(other)) if other.size else 0.0
    confidence = (energy - second) / max(energy, 1e-9)

    if APPLY_MIN_ENERGY_CUT and energy < MIN_EVENT_ENERGY_KEV:
        return False, target_contact, energy, confidence
    if APPLY_MAX_ENERGY_CUT and energy > MAX_EVENT_ENERGY_KEV:
        return False, target_contact, energy, confidence
    if APPLY_FINAL_CONFIDENCE_CUT and confidence < MIN_FINAL_CONFIDENCE:
        return False, target_contact, energy, confidence
    return True, target_contact, energy, confidence


# ============================ Inference ============================

@dataclass
class Prediction:
    event_index: int
    target_contact: int
    waveform_energy_keV: float
    final_confidence: float
    core_crossing_sample: float
    alignment_shift_samples: float
    x_mm: float
    y_mm: float
    z_mm: float
    sigma_x_mm: float
    sigma_y_mm: float
    sigma_z_mm: float
    sigma_3d_proxy_mm: float


def get_checkpoint_array(checkpoint: dict, key: str) -> np.ndarray:
    if key not in checkpoint:
        raise KeyError(f"Checkpoint is missing required key: {key}")
    return np.asarray(checkpoint[key], dtype=np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not REAL_WAVEFORM_FILE.is_file():
        raise FileNotFoundError(REAL_WAVEFORM_FILE)
    if not MODEL_CHECKPOINT.is_file():
        raise FileNotFoundError(MODEL_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    pool_bins = int(config.get("pool_bins", POOL_BINS))
    dropout = float(config.get("dropout", DROPOUT))
    min_log_variance = float(config.get("minimum_log_variance", -10.0))
    max_log_variance = float(config.get("maximum_log_variance", 5.0))

    model = TwoStageModel(dropout=dropout, pool_bins=pool_bins).to(device)
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    model.coordinate_head.load_state_dict(checkpoint["coordinate_head_state_dict"])
    model.uncertainty_head.load_state_dict(checkpoint["uncertainty_head_state_dict"])
    model.eval()

    channel_mean = get_checkpoint_array(checkpoint, "channel_mean_keV").reshape(9, 1)
    channel_std = np.maximum(
        get_checkpoint_array(checkpoint, "channel_std_keV").reshape(9, 1), 1e-6
    )
    position_mean = get_checkpoint_array(checkpoint, "position_mean_mm").reshape(1, 3)
    position_std = np.maximum(
        get_checkpoint_array(checkpoint, "position_std_mm").reshape(1, 3), 1e-6
    )

    calibration_scale = np.asarray(
        checkpoint.get(
            "uncertainty_calibration_scale",
            FALLBACK_UNCERTAINTY_CALIBRATION_SCALE,
        ),
        dtype=np.float32,
    ).reshape(1, 3)

    print(f"Device: {device}")
    print(f"Pool bins: {pool_bins}")
    print(f"Uncertainty calibration scale: {calibration_scale.ravel()}")
    print(f"Reading real waveforms: {REAL_WAVEFORM_FILE}")

    raw_events = read_real_events(REAL_WAVEFORM_FILE)
    print(f"Complete parsed events: {len(raw_events)}")

    accepted_waveforms: List[np.ndarray] = []
    accepted_metadata: List[Tuple[int, int, float, float, float, float]] = []

    for event_index, raw in enumerate(raw_events, start=1):
        waveform = baseline_subtract_and_calibrate(raw)
        waveform, crossing, shift = align_waveform(waveform)
        if ALIGN_REAL_EVENTS and not np.isfinite(shift):
            continue

        accepted, target, energy, confidence = event_selection(waveform)
        if not accepted:
            continue

        standardized = (waveform - channel_mean) / channel_std
        if not np.all(np.isfinite(standardized)):
            continue

        accepted_waveforms.append(standardized.astype(np.float32))
        accepted_metadata.append(
            (event_index, target, energy, confidence, crossing, shift)
        )

    if not accepted_waveforms:
        raise RuntimeError("No real events survived preprocessing and selection")

    inputs = np.stack(accepted_waveforms, axis=0)
    predictions: List[Prediction] = []
    batch_size = 512

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(inputs.shape[0], start + batch_size)
            batch = torch.from_numpy(inputs[start:stop]).to(device)
            mean_normalized, log_variance_normalized = model(batch)

            mean_normalized = mean_normalized.cpu().numpy()
            log_variance_normalized = np.clip(
                log_variance_normalized.cpu().numpy(),
                min_log_variance,
                max_log_variance,
            )

            coordinates_mm = mean_normalized * position_std + position_mean
            sigma_raw_mm = (
                np.exp(0.5 * log_variance_normalized) * position_std
            )
            sigma_calibrated_mm = sigma_raw_mm * calibration_scale

            for local_index in range(stop - start):
                metadata = accepted_metadata[start + local_index]
                coordinate = coordinates_mm[local_index]
                sigma = sigma_calibrated_mm[local_index]
                sigma_3d = float(np.sqrt(np.sum(sigma**2)))

                predictions.append(
                    Prediction(
                        event_index=int(metadata[0]),
                        target_contact=int(metadata[1]),
                        waveform_energy_keV=float(metadata[2]),
                        final_confidence=float(metadata[3]),
                        core_crossing_sample=float(metadata[4]),
                        alignment_shift_samples=float(metadata[5]),
                        x_mm=float(coordinate[0]),
                        y_mm=float(coordinate[1]),
                        z_mm=float(coordinate[2]),
                        sigma_x_mm=float(sigma[0]),
                        sigma_y_mm=float(sigma[1]),
                        sigma_z_mm=float(sigma[2]),
                        sigma_3d_proxy_mm=sigma_3d,
                    )
                )

    write_predictions_csv(predictions)
    create_plots(predictions)

    uncertainty = np.array(
        [prediction.sigma_3d_proxy_mm for prediction in predictions]
    )
    print(f"Classified events: {len(predictions)}")
    print(f"Median 3D uncertainty proxy: {np.median(uncertainty):.3f} mm")
    print(f"90th-percentile uncertainty proxy: {np.percentile(uncertainty, 90):.3f} mm")
    print(f"Saved outputs to: {OUTPUT_DIR}")


def write_predictions_csv(predictions: List[Prediction]) -> None:
    output_path = OUTPUT_DIR / "real_waveform_predictions.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "event_index",
            "target_contact",
            "waveform_energy_keV",
            "final_confidence",
            "core_crossing_sample",
            "alignment_shift_samples",
            "pred_x_mm",
            "pred_y_mm",
            "pred_z_ssd_mm",
            "sigma_x_mm",
            "sigma_y_mm",
            "sigma_z_mm",
            "sigma_3d_proxy_mm",
        ])
        for p in predictions:
            writer.writerow([
                p.event_index,
                p.target_contact,
                p.waveform_energy_keV,
                p.final_confidence,
                p.core_crossing_sample,
                p.alignment_shift_samples,
                p.x_mm,
                p.y_mm,
                p.z_mm,
                p.sigma_x_mm,
                p.sigma_y_mm,
                p.sigma_z_mm,
                p.sigma_3d_proxy_mm,
            ])


def create_plots(predictions: List[Prediction]) -> None:
    x = np.array([p.x_mm for p in predictions], dtype=np.float32)
    y = np.array([p.y_mm for p in predictions], dtype=np.float32)
    z = np.array([p.z_mm for p in predictions], dtype=np.float32)
    sigma3d = np.array(
        [p.sigma_3d_proxy_mm for p in predictions], dtype=np.float32
    )

    # Full uncertainty color range.
    fig, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        x,
        y,
        c=sigma3d,
        s=MARKER_SIZE,
        alpha=MARKER_ALPHA,
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="Predicted 3D uncertainty proxy (mm)")
    axis.set_xlabel("Predicted x (mm)")
    axis.set_ylabel("Predicted y (mm)")
    axis.set_title("Real HPGe Hits Colored by Predicted Uncertainty")
    axis.set_xlim(-100, 100)
    axis.set_ylim(-100, 100)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_hits_xy_uncertainty.png", dpi=300)
    plt.close(fig)

    # Clipped color range makes ordinary uncertainty structure visible without
    # a few uncertain outliers controlling the full color scale.
    color_max = float(np.percentile(sigma3d, UNCERTAINTY_CLIP_PERCENTILE))
    fig, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        x,
        y,
        c=np.minimum(sigma3d, color_max),
        s=MARKER_SIZE,
        alpha=MARKER_ALPHA,
        cmap="viridis",
        vmin=0.0,
        vmax=color_max,
        linewidths=0,
    )
    fig.colorbar(
        scatter,
        ax=axis,
        label=(
            "Predicted 3D uncertainty proxy (mm), "
            f"clipped at p{UNCERTAINTY_CLIP_PERCENTILE:g}"
        ),
    )
    axis.set_xlabel("Predicted x (mm)")
    axis.set_ylabel("Predicted y (mm)")
    axis.set_title("Real HPGe Hits Colored by Predicted Uncertainty")
    axis.set_xlim(-100, 100)
    axis.set_ylim(-100, 100)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_hits_xy_uncertainty_clipped.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(sigma3d, bins=60)
    axis.set_xlabel("Predicted 3D uncertainty proxy (mm)")
    axis.set_ylabel("Events")
    axis.set_title("Predicted Uncertainty Distribution for Real Events")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_uncertainty_histogram.png", dpi=300)
    plt.close(fig)

    # Optional predicted depth distribution for sanity checking.
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(z, bins=60)
    axis.set_xlabel("Predicted z in SSD coordinates (mm)")
    axis.set_ylabel("Events")
    axis.set_title("Predicted Real-Event Depth Distribution")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_predicted_z_histogram.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()