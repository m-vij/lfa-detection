"""Calibration: where do the six bands sit, and what are the training pairs?

Step 01 measures each band's position (cm, relative to the control peak) on the
calibration images and stores the mean / std in ``calibration.json``.
Step 02 then uses those positions to extract (control, signal) intensity pairs
for every concentration, producing one training table per band.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data_check import list_images
from .detection import (
    DetectionError,
    calculate_real_world_distance,
    locate_strip,
)
from .signal_processing import sample_signal, sample_signal_using_distance

log = logging.getLogger(__name__)

NUM_BANDS = 6


# ---------------------------------------------------------------------------
# Step 01: band positions
# ---------------------------------------------------------------------------
def measure_band_positions(yolo_model, class_names, calibration_dir, concentrations, cfg):
    """Measure each band's distance from the control peak on every calibration image.

    Returns ``{band: [position_cm, ...]}`` for bands 1..6. Images that fail
    detection or do not show exactly 7 peaks are skipped (and logged).
    """
    sig = cfg["signal"]
    rw = cfg["geometry"]["real_world_distance_cm"]
    positions = {b: [] for b in range(1, NUM_BANDS + 1)}

    for concentration in concentrations:
        base_dir = Path(calibration_dir) / str(concentration)
        log.info("Measuring band positions in %s", base_dir)
        files = list_images(base_dir)
        results = yolo_model(files)

        for path, result in zip(files, results):
            try:
                midpoint, slope, distance, target = locate_strip(result, class_names)
                img = cv2.imread(path)
                peak_locations, _ = sample_signal(
                    img, midpoint, target, slope, distance,
                    track_width=sig["track_width"], tolerance=sig["tolerance"],
                    prominence_frac=sig["first_pass_prominence"],
                    peak_width=sig["peak_width"],
                    savgol_window=sig["savgol_window"],
                    savgol_polyorder=sig["savgol_polyorder"],
                    baseline_sigma=sig["baseline_sigma"],
                )
                if len(peak_locations) != 7:
                    raise DetectionError(f"Expected 7 peaks, found {len(peak_locations)}")

                control = peak_locations[0]
                for band in range(1, NUM_BANDS + 1):
                    px = peak_locations[band] - control
                    positions[band].append(calculate_real_world_distance(distance, px, rw))
            except (DetectionError, ZeroDivisionError) as err:
                log.warning("Skipped %s: %s", path, err)

    return positions


def summarize_band_positions(positions):
    """Mean / (population) std of each band's position in cm."""
    return {
        "mean": [float(np.mean(positions[b])) for b in range(1, NUM_BANDS + 1)],
        "std": [float(np.std(positions[b])) for b in range(1, NUM_BANDS + 1)],
        "n_images": len(positions[1]),
    }


def save_calibration(path, summary, rw_distance_cm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "band_position_cm_mean": summary["mean"],
        "band_position_cm_std": summary["std"],
        "n_images": summary["n_images"],
        "real_world_distance_cm": rw_distance_cm,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path):
    """Return ``(band_means_cm, band_stds_cm)`` as lists of six floats."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Run scripts/01_calibrate_bands.py first."
        )
    with open(path) as f:
        payload = json.load(f)
    return payload["band_position_cm_mean"], payload["band_position_cm_std"]


# ---------------------------------------------------------------------------
# Step 02: training pairs
# ---------------------------------------------------------------------------
def extract_training_pairs(yolo_model, class_names, calibration_dir, concentrations,
                           band_means, band_stds, cfg):
    """Extract (control, signal) intensities for each band at each concentration.

    Returns a dict ``{band: DataFrame[Control, Signal, Target]}`` where
    ``Target`` is the concentration.
    """
    sig = cfg["signal"]
    rw = cfg["geometry"]["real_world_distance_cm"]
    rows = {b: [] for b in range(1, NUM_BANDS + 1)}

    for concentration in concentrations:
        base_dir = Path(calibration_dir) / str(concentration)
        log.info("Extracting intensities in %s", base_dir)
        files = list_images(base_dir)
        results = yolo_model(files)

        for path, result in zip(files, results):
            try:
                midpoint, slope, distance, target = locate_strip(result, class_names)
                img = cv2.imread(path)
                _, intensities = sample_signal_using_distance(
                    img, midpoint, target, slope, distance,
                    list(band_means), list(band_stds),
                    track_width=sig["track_width"], tolerance=sig["tolerance"],
                    rw_distance=rw,
                    control_prominence=sig["control_prominence"],
                    peak_width=sig["peak_width"],
                    savgol_window=sig["savgol_window"],
                    savgol_polyorder=sig["savgol_polyorder"],
                    baseline_sigma=sig["baseline_sigma"],
                )
                for band in range(1, NUM_BANDS + 1):
                    rows[band].append((intensities[0], intensities[band], concentration))
            except (DetectionError, ValueError, ZeroDivisionError) as err:
                log.warning("Skipped %s: %s", path, err)

    return {
        band: pd.DataFrame(r, columns=["Control", "Signal", "Target"])
        for band, r in rows.items()
    }
