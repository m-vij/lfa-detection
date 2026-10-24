"""Intensity profile extraction along a test strip.

Two extraction passes exist:

``sample_signal``
    First pass used during *calibration* (step 01). Finds every peak on the
    profile with ``scipy.signal.find_peaks``; used to measure where the six
    bands sit relative to the control peak, in centimetres.

``sample_signal_using_distance``
    Second pass, used identically for training data (step 02) and for patient
    images (step 04). Locates the control peak, then takes the **maximum signal
    inside a +/- 3 sigma window** around each band's calibrated position.

Training and patient images deliberately share this single implementation so
the regressors always see intensities computed the same way.

Note: the track is averaged over ``arange(-(w//2), w//2)`` columns but divided
by ``w``. For an odd ``track_width`` (e.g. 15) that is 14 columns / 15, i.e.
intensities are scaled by 14/15. This is the behaviour the existing models were
trained with; changing it requires retraining steps 02-05.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def _sampling_direction(midpoint, target, slope):
    """Unit vector along the strip, oriented from ``midpoint`` toward ``target``."""
    angle = math.atan(slope)
    vx, vy = math.cos(angle), math.sin(angle)

    dx = target[0] - midpoint[0]
    dy = target[1] - midpoint[1]
    if vx * dx + vy * dy < 0:
        angle += math.pi
        vx, vy = math.cos(angle), math.sin(angle)
    return angle, vx, vy


def _track_offsets(track_width):
    """Perpendicular pixel offsets for averaging across the strip width."""
    offsets = np.arange(-(track_width // 2), track_width // 2)
    return offsets, track_width


def sample_track(img, midpoint, target, slope, distance, tolerance, track_width,
                 min_steps=0):
    """Sample the (green channel) image along the strip axis.

    Returns ``(avg_signal, x_path, y_path)`` where ``avg_signal`` is the raw
    intensity averaged across the track width at each step along the path.
    """
    gray = img[:, :, 1]
    h, w = gray.shape

    angle, vx, vy = _sampling_direction(midpoint, target, slope)

    x_start = midpoint[0] - tolerance * vx
    y_start = midpoint[1] - tolerance * vy
    x_end = midpoint[0] + (distance + tolerance) * vx
    y_end = midpoint[1] + (distance + tolerance) * vy

    num_steps = int(abs(distance) + 2 * tolerance)
    if min_steps:
        num_steps = max(min_steps, num_steps)
    x_path = np.linspace(x_start, x_end, num_steps)
    y_path = np.linspace(y_start, y_end, num_steps)

    nx, ny = -math.sin(angle), math.cos(angle)
    offsets, divisor = _track_offsets(track_width)

    signal_sum = np.zeros(num_steps, dtype=float)
    for offset in offsets:
        sample_x = np.clip((x_path + offset * nx).astype(int), 0, w - 1)
        sample_y = np.clip((y_path + offset * ny).astype(int), 0, h - 1)
        signal_sum += gray[sample_y, sample_x]

    return signal_sum / divisor, x_path, y_path


def clean_signal(avg_signal, savgol_window=21, savgol_polyorder=2, baseline_sigma=20):
    """Invert, smooth and baseline-correct a raw intensity profile.

    Dark bands become positive peaks; slow background drift is removed with a
    wide gaussian baseline.
    """
    inverted = np.max(avg_signal) - avg_signal
    win = min(savgol_window, len(inverted) // 2 * 2 - 1)
    smooth = savgol_filter(inverted, win, savgol_polyorder) if win >= 3 else inverted
    baseline = gaussian_filter(smooth, sigma=baseline_sigma)
    return np.maximum(smooth - baseline, 0)


# ---------------------------------------------------------------------------
# Pass 1: find every peak (calibration geometry)
# ---------------------------------------------------------------------------
def sample_signal(img, midpoint, target, slope, distance, tolerance=5, track_width=30,
                  prominence_frac=0.02, peak_width=4, savgol_window=21,
                  savgol_polyorder=2, baseline_sigma=20, show=False):
    """Detect all peaks along the strip.

    Returns ``(peak_indices, peak_intensities)``. For a valid calibration image
    there should be exactly 7 peaks: the control plus six bands.
    """
    avg_signal, x_path, y_path = sample_track(
        img, midpoint, target, slope, distance, tolerance, track_width,
    )
    final_signal = clean_signal(avg_signal, savgol_window, savgol_polyorder, baseline_sigma)

    peaks, _ = find_peaks(
        final_signal, prominence=np.max(final_signal) * prominence_frac, width=peak_width
    )

    if show:
        from .visualization import plot_first_pass

        plot_first_pass(img, x_path, y_path, midpoint, slope, final_signal, peaks)

    return peaks, final_signal[peaks]


# ---------------------------------------------------------------------------
# Pass 2: locate bands from calibrated distances
# ---------------------------------------------------------------------------
def sample_signal_using_distance(img, midpoint, target, slope, distance,
                                 real_world_peak_locations, real_world_peak_std,
                                 tolerance=30, track_width=15, rw_distance=2.55,
                                 control_prominence=0.05, peak_width=4,
                                 savgol_window=21, savgol_polyorder=2,
                                 baseline_sigma=20, show=False):
    """Quantify the control and six band intensities using known band distances.

    Each band's intensity is the maximum processed signal inside a +/- 3 sigma
    window around its calibrated position (relative to the control peak).

    Parameters
    ----------
    real_world_peak_locations, real_world_peak_std
        Mean / std of each band's position (cm) relative to the control peak,
        from ``lfa.calibration``.

    Returns
    -------
    peak_locations : int array (7,)   pixel index of control + 6 bands
    peak_intensities : float array (7,)   processed intensity of control + 6 bands
    """
    if distance <= 0:
        raise ValueError("Calibration distance must be positive.")

    avg_signal, x_path, y_path = sample_track(
        img, midpoint, target, slope, distance, tolerance, track_width, min_steps=1,
    )
    final_signal = clean_signal(avg_signal, savgol_window, savgol_polyorder, baseline_sigma)

    cm_per_pixel = rw_distance / distance
    n = len(final_signal)

    # Control peak = first prominent peak from the left.
    peaks, _ = find_peaks(
        final_signal,
        prominence=max(float(np.max(final_signal)) * control_prominence, 1e-8),
        width=peak_width,
    )
    if len(peaks) == 0:
        raise ValueError("No control peak detected.")

    control_idx = int(peaks[0])
    peak_locations = [control_idx]
    peak_intensities = [float(final_signal[control_idx])]
    windows = []

    for location_cm, std_cm in zip(real_world_peak_locations, real_world_peak_std):
        expected_idx = control_idx + location_cm / cm_per_pixel
        std_px = std_cm / cm_per_pixel

        window_start = max(0, int(np.floor(expected_idx - 3 * std_px)))
        window_end = min(n - 1, int(np.ceil(expected_idx + 3 * std_px)))
        windows.append((expected_idx - 3 * std_px, expected_idx + 3 * std_px))

        if window_start <= window_end:
            # Band intensity = maximum signal inside the window.
            window = final_signal[window_start:window_end + 1]
            peak_idx = window_start + int(np.argmax(window))
            peak_intensity = float(final_signal[peak_idx])
        else:  # window fell entirely outside the signal
            peak_idx = int(np.clip(round(expected_idx), 0, n - 1))
            peak_intensity = 0.0

        peak_locations.append(peak_idx)
        peak_intensities.append(peak_intensity)

    peak_locations = np.asarray(peak_locations, dtype=int)
    peak_intensities = np.asarray(peak_intensities, dtype=float)

    if show:
        from .visualization import plot_band_extraction

        plot_band_extraction(
            img, x_path, y_path, midpoint, slope, final_signal,
            peak_locations, peak_intensities, windows,
        )

    return peak_locations, peak_intensities
