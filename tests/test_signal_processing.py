import numpy as np
import pytest

from lfa.signal_processing import sample_signal, sample_signal_using_distance

LOCS_CM = [0.40, 0.80, 1.20, 1.60, 2.00, 2.40]  # 55 px steps on the synthetic strip
STD_CM = [0.02] * 6


def test_first_pass_finds_seven_peaks(strip_image, geometry):
    midpoint, slope, distance, target = geometry
    peaks, intensities = sample_signal(strip_image, midpoint, target, slope, distance,
                                       track_width=15, tolerance=30)
    assert len(peaks) == 7
    assert np.all(np.diff(peaks) > 0)
    assert np.all(intensities > 0)


def test_distance_pass_recovers_bands(strip_image, geometry):
    midpoint, slope, distance, target = geometry
    locs, ints = sample_signal_using_distance(strip_image, midpoint, target, slope, distance,
                                              LOCS_CM, STD_CM)
    assert len(locs) == len(ints) == 7
    # bands are ~55 px apart along the path
    assert np.allclose(np.diff(locs), 55, atol=3)
    # synthetic amplitudes decrease from control to band 6
    assert np.all(np.diff(ints) < 0)


def test_band_intensity_is_window_maximum(strip_image, geometry):
    """Each band intensity must equal the max of the processed signal inside its window."""
    from lfa.signal_processing import clean_signal, sample_track

    midpoint, slope, distance, target = geometry
    locs, ints = sample_signal_using_distance(strip_image, midpoint, target, slope, distance,
                                              LOCS_CM, STD_CM, tolerance=30, track_width=15)
    avg, _, _ = sample_track(strip_image, midpoint, target, slope, distance, 30, 15, min_steps=1)
    final = clean_signal(avg)
    for loc, val in zip(locs, ints):
        assert val == final[loc]
    assert ints[1] == final[locs[0] + 40:locs[0] + 70].max()  # band 1 sits ~55 px after control


def test_no_mode_argument(strip_image, geometry):
    """Training and patient images share one implementation."""
    midpoint, slope, distance, target = geometry
    with pytest.raises(TypeError):
        sample_signal_using_distance(strip_image, midpoint, target, slope, distance,
                                     LOCS_CM, STD_CM, mode="inference")


def test_invalid_distance(strip_image, geometry):
    midpoint, slope, distance, target = geometry
    with pytest.raises(ValueError):
        sample_signal_using_distance(strip_image, midpoint, target, slope, 0,
                                     LOCS_CM, STD_CM)
