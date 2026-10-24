"""Shared fixtures: a synthetic strip image and a fake YOLO model (no weights needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Geometry of the synthetic strip (pixels)
BAND_X = [135, 190, 245, 300, 355, 410, 465]  # control + 6 bands, 55 px apart
C1 = [100, 230, 10, 10]
C2 = [100, 270, 10, 10]
ONE = [450, 250, 10, 10]


def make_strip_image(seed=0, amplitudes=(90, 60, 55, 50, 40, 35, 30)):
    rng = np.random.default_rng(seed)
    H, W = 500, 800
    g = np.full((H, W), 200.0)
    xs = np.arange(W)
    rows = np.abs(np.arange(H)[:, None] - 250) < 40
    for x0, a in zip(BAND_X, amplitudes):
        g -= a * np.exp(-((xs[None, :] - x0) ** 2) / (2 * 4.0 ** 2)) * rows
    g += rng.normal(0, 1.5, g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Vec:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)


class _Box:
    def __init__(self, cls_id, xywh):
        self.cls = [_Scalar(cls_id)]
        self.xywh = [_Vec(xywh)]


class _Result:
    def __init__(self):
        self.boxes = [_Box(0, C1), _Box(0, C2), _Box(1, ONE)]


class FakeYolo:
    """Mimics ``ultralytics.YOLO.__call__``: one result (C, C, 1) per input path."""

    names = {0: "C", 1: "1"}

    def __call__(self, files):
        if isinstance(files, (str, Path)):
            files = [files]
        return [_Result() for _ in files]


@pytest.fixture
def strip_image():
    return make_strip_image()


@pytest.fixture
def fake_yolo():
    return FakeYolo()


@pytest.fixture
def geometry():
    from lfa.detection import roi_slope_center

    return roi_slope_center([C1, C2], [ONE])
