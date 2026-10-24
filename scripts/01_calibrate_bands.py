#!/usr/bin/env python
"""Step 01 -- measure where the six bands sit relative to the control peak.

Reads   data/calibration/{1..6}/*.jpg  and  models/best.pt
Writes  models/calibration.json         (band position mean/std in cm)
        results/calibration/band_positions_raw.csv
"""

import argparse
import logging

import pandas as pd

from lfa.calibration import measure_band_positions, save_calibration, summarize_band_positions
from lfa.config import ensure_dir, load_config, resolve_path, setup_logging
from lfa.detection import load_yolo

log = logging.getLogger("01_calibrate_bands")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    yolo, class_names = load_yolo(resolve_path(cfg, "yolo_weights"))
    positions = measure_band_positions(
        yolo, class_names, resolve_path(cfg, "calibration_dir"), cfg["concentrations"], cfg
    )

    summary = summarize_band_positions(positions)
    if summary["n_images"] == 0:
        raise SystemExit("No calibration image produced 7 peaks -- check data and YOLO weights.")

    for band in range(1, 7):
        log.info("Band %d  mean=%.4f cm  std=%.4f cm", band,
                 summary["mean"][band - 1], summary["std"][band - 1])
    log.info("Used %d calibration images", summary["n_images"])

    save_calibration(resolve_path(cfg, "calibration_json"), summary,
                     cfg["geometry"]["real_world_distance_cm"])

    out_dir = ensure_dir(resolve_path(cfg, "results_dir") / "calibration")
    pd.DataFrame({f"band_{b}_cm": v for b, v in positions.items()}).to_csv(
        out_dir / "band_positions_raw.csv", index=False)
    log.info("Saved calibration to %s", resolve_path(cfg, "calibration_json"))


if __name__ == "__main__":
    main()
