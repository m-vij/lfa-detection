#!/usr/bin/env python
"""Debug helper -- run the pipeline on ONE image and show the sampling path + band peaks.

Example:
    python scripts/inspect_image.py data/patient_samples/IMG_0001.jpg
    python scripts/inspect_image.py IMG.jpg --save results/inspect.png
"""

import argparse

import cv2

from lfa.calibration import load_calibration
from lfa.config import load_config, resolve_path
from lfa.detection import load_yolo, locate_strip
from lfa.signal_processing import sample_signal_using_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image", help="Path to the strip image")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sig = cfg["signal"]
    band_means, band_stds = load_calibration(resolve_path(cfg, "calibration_json"))
    yolo, class_names = load_yolo(resolve_path(cfg, "yolo_weights"))

    result = yolo(args.image)[0]
    midpoint, slope, distance, target = locate_strip(result, class_names)
    img = cv2.imread(args.image)

    _, intensities = sample_signal_using_distance(
        img, midpoint, target, slope, distance, band_means, band_stds,
        track_width=sig["track_width"], tolerance=sig["tolerance"],
        rw_distance=cfg["geometry"]["real_world_distance_cm"],
        control_prominence=sig["control_prominence"], peak_width=sig["peak_width"],
        show=True,
    )

    print(f"Image: {args.image}")
    print(f"Control intensity: {intensities[0]:.3f}")
    for i in range(1, 7):
        print(f"Band {i} intensity: {intensities[i]:.3f} | ratio to control: {intensities[i] / intensities[0]:.4f}")


if __name__ == "__main__":
    main()
