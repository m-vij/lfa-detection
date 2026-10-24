#!/usr/bin/env python
"""Step 03 -- train one concentration regressor per band.

Reads   results/datasets/band{1..6}.csv
Writes  models/final_model_band{1..6}.pth
        results/regressors/cv_metrics.csv
        results/regressors/plots/band{N}_*.png
"""

import argparse
import logging

import pandas as pd

from lfa.config import ensure_dir, load_config, resolve_path, setup_logging
from lfa.models import band_model_path, save_checkpoint, train_regressor_cv

log = logging.getLogger("03_train_regressors")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--bands", type=int, nargs="+", default=list(range(1, 7)),
                        help="Which bands to train (default: all)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    reg = cfg["regressor"]

    results_dir = resolve_path(cfg, "results_dir")
    data_dir = results_dir / "datasets"
    out_dir = ensure_dir(results_dir / "regressors")
    plot_dir = ensure_dir(out_dir / "plots")
    models_dir = ensure_dir(resolve_path(cfg, "models_dir"))

    summary = []
    for band in args.bands:
        log.info("=== Training band %d ===", band)
        df = pd.read_csv(data_dir / f"band{band}.csv")

        results = train_regressor_cv(
            df, n_splits=reg["n_splits"], epochs=reg["epochs"], batch_size=reg["batch_size"],
            lr=reg["lr"], huber_delta=reg["huber_delta"], seed=reg["seed"],
            plot_dir=plot_dir, plot_prefix=f"band{band}",
        )

        save_checkpoint(band_model_path(models_dir, band), results)
        log.info("Saved %s", band_model_path(models_dir, band))

        fr = results["fold_results"]
        summary.append({
            "band": band, "best_fold": results["best_fold"],
            "MAE_mean": fr[:, 0].mean(), "MAE_std": fr[:, 0].std(),
            "RMSE_mean": fr[:, 1].mean(), "RMSE_std": fr[:, 1].std(),
            "R2_mean": fr[:, 2].mean(), "R2_std": fr[:, 2].std(),
        })

    pd.DataFrame(summary).to_csv(out_dir / "cv_metrics.csv", index=False)
    log.info("CV metrics -> %s", out_dir / "cv_metrics.csv")


if __name__ == "__main__":
    main()
