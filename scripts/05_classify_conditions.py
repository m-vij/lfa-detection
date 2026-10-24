#!/usr/bin/env python
"""Step 05 -- classify patient condition from the six predicted concentrations.

Reads   results/patient_results/patient_predictions.csv
Writes  results/patient_results/cv5/
            fold_metrics.csv, oof_predictions.csv, oof_confusion_matrix.png,
            misclassified_images.csv, misclassified_images.png,
            oof_embeddings.npy, (optional) umap_embeddings.png
"""

import argparse
import logging

import numpy as np

from lfa.classification import (build_prediction_table, cross_validate_classifier,
                                find_misclassified, load_classification_data)
from lfa.config import ensure_dir, image_search_dirs, load_config, resolve_path, setup_logging
from lfa.visualization import plot_confusion_matrix, plot_misclassified, plot_umap_embeddings

log = logging.getLogger("05_classify_conditions")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--umap", action="store_true",
                        help="Also plot a UMAP of held-out embeddings (needs umap-learn)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    clf = cfg["classifier"]

    patient_dir = resolve_path(cfg, "results_dir") / "patient_results"
    out_dir = ensure_dir(patient_dir / "cv5")

    data, X, y, encoder = load_classification_data(
        patient_dir / "patient_predictions.csv",
        condition_aliases=clf.get("condition_aliases"),
    )
    classes = encoder.classes_
    log.info("Samples: %d | Classes: %s", len(X), dict(zip(classes, np.bincount(y))))

    cv = cross_validate_classifier(
        X, y, classes, n_splits=clf["n_splits"], epochs=clf["epochs"], lr=clf["lr"],
        weight_decay=clf["weight_decay"], dropout=clf["dropout"], seed=clf["seed"],
    )

    cv["fold_metrics"].to_csv(out_dir / "fold_metrics.csv", index=False)
    build_prediction_table(data, y, cv["oof_predictions"], cv["oof_probabilities"], encoder).to_csv(
        out_dir / "oof_predictions.csv", index=False)
    np.save(out_dir / "oof_embeddings.npy", cv["oof_embeddings"])

    plot_confusion_matrix(cv["confusion_matrix"], classes, out_dir / "oof_confusion_matrix.png")

    error_df, misclassified = find_misclassified(data, y, cv["oof_predictions"], encoder)
    log.info("Total: %d | Correct: %d | Incorrect: %d | Error rate: %.2f%%",
             len(error_df), error_df["Correct"].sum(), len(misclassified),
             100 * (~error_df["Correct"]).mean())
    misclassified.to_csv(out_dir / "misclassified_images.csv", index=False)
    plot_misclassified(misclassified, image_search_dirs(cfg),
                       out_dir / "misclassified_images.png")

    if args.umap:
        plot_umap_embeddings(cv["oof_embeddings"], y, classes, out_dir / "umap_embeddings.png",
                             random_state=clf["seed"])

    log.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
