"""End-to-end check of steps 01-02 using synthetic images and a fake YOLO."""

import cv2
import numpy as np
import pytest

from lfa.calibration import (extract_training_pairs, load_calibration, measure_band_positions,
                             save_calibration, summarize_band_positions)
from lfa.config import load_config

from conftest import make_strip_image


@pytest.fixture
def calibration_dir(tmp_path):
    # Band 1 gets darker with concentration so the target is learnable.
    for conc in range(1, 7):
        folder = tmp_path / str(conc)
        folder.mkdir()
        for k in range(2):
            amps = (90, 10 * conc, 55, 50, 40, 35, 30)
            cv2.imwrite(str(folder / f"img{k}.png"), make_strip_image(seed=conc * 10 + k, amplitudes=amps))
        (folder / ".DS_Store").write_text("junk")  # hidden files must be ignored
    return tmp_path


def test_band_positions_roundtrip(calibration_dir, fake_yolo, tmp_path):
    cfg = load_config()
    positions = measure_band_positions(fake_yolo, fake_yolo.names, calibration_dir,
                                       cfg["concentrations"], cfg)
    summary = summarize_band_positions(positions)

    assert summary["n_images"] == 12
    assert summary["mean"] == pytest.approx([0.40, 0.80, 1.20, 1.60, 2.00, 2.40], abs=0.03)

    path = tmp_path / "calibration.json"
    save_calibration(path, summary, 2.55)
    means, stds = load_calibration(path)
    assert means == summary["mean"] and stds == summary["std"]


def test_training_pairs(calibration_dir, fake_yolo):
    cfg = load_config()
    means = [0.40, 0.80, 1.20, 1.60, 2.00, 2.40]
    stds = [0.02] * 6
    tables = extract_training_pairs(fake_yolo, fake_yolo.names, calibration_dir,
                                    cfg["concentrations"], means, stds, cfg)

    assert set(tables) == set(range(1, 7))
    band1 = tables[1]
    assert list(band1.columns) == ["Control", "Signal", "Target"]
    assert len(band1) == 12
    # Band 1 signal grows with concentration
    by_conc = band1.groupby("Target")["Signal"].mean()
    assert by_conc.is_monotonic_increasing
