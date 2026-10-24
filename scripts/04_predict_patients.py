#!/usr/bin/env python
"""Step 04 -- predict band concentrations for every patient image.

Reads   data/patient_readings.csv + data/patient_samples/, plus EVERY image in
        data/positive_control/ (label "positive") and data/negative_control/ (label
        "negative"), models/best.pt,
        models/calibration.json, models/final_model_band{1..6}.pth
Writes  results/patient_results/patient_predictions.csv
        results/patient_results/diagnostic_plots/*.png
"""

import argparse
import logging

from lfa.config import (ensure_dir, image_search_dirs, labeled_image_dirs, load_config,
                        resolve_path, setup_logging)
from lfa.inference import PatientPipeline

log = logging.getLogger("04_predict_patients")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    out_dir = ensure_dir(resolve_path(cfg, "results_dir") / "patient_results")
    pipeline = PatientPipeline(
        yolo_weights=resolve_path(cfg, "yolo_weights"),
        calibration_json=resolve_path(cfg, "calibration_json"),
        models_dir=resolve_path(cfg, "models_dir"),
        cfg=cfg,
        plot_dir=out_dir / "diagnostic_plots",
    )

    results = pipeline.run(resolve_path(cfg, "patient_csv"), image_search_dirs(cfg),
                           labeled_dirs=labeled_image_dirs(cfg))

    path = out_dir / "patient_predictions.csv"
    results.to_csv(path, index=False)
    log.info("Processing complete. Results saved to %s", path)
    print(results["Status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
