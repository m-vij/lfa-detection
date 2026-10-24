#!/usr/bin/env python
"""Step 02 -- extract (control, signal) intensity pairs for every concentration.

Reads   data/calibration/{1..6}/*.jpg, models/best.pt, models/calibration.json
Writes  results/datasets/band{1..6}.csv   columns: Control, Signal, Target
"""

import argparse
import logging

from lfa.calibration import extract_training_pairs, load_calibration
from lfa.config import ensure_dir, load_config, resolve_path, setup_logging
from lfa.detection import load_yolo

log = logging.getLogger("02_build_dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    band_means, band_stds = load_calibration(resolve_path(cfg, "calibration_json"))
    yolo, class_names = load_yolo(resolve_path(cfg, "yolo_weights"))

    datasets = extract_training_pairs(
        yolo, class_names, resolve_path(cfg, "calibration_dir"), cfg["concentrations"],
        band_means, band_stds, cfg,
    )

    out_dir = ensure_dir(resolve_path(cfg, "results_dir") / "datasets")
    for band, df in datasets.items():
        df.to_csv(out_dir / f"band{band}.csv", index=False)
        log.info("Band %d: %d rows -> %s", band, len(df), out_dir / f"band{band}.csv")


if __name__ == "__main__":
    main()
