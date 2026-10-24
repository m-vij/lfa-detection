#!/usr/bin/env python
"""Check where every image in patient_readings.csv is found -- run this BEFORE step 04.

Builds the full list of images to analyse (CSV rows plus every image in the labeled
control folders), then reports missing files, filenames found in more than one
folder, and a Condition x folder table.

Writes  results/data_check.csv
"""

import argparse

from lfa.config import ensure_dir, image_search_dirs, labeled_image_dirs, load_config, resolve_path
from lfa.data_check import check_image_lookup, summarize_lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = image_search_dirs(cfg)
    for d in dirs:
        print(f"{'OK     ' if d.is_dir() else 'MISSING'} {d}")
    print()

    labeled = labeled_image_dirs(cfg)
    for d, label in labeled.items():
        print(f"{'OK     ' if d.is_dir() else 'MISSING'} {d}  (label: {label})")
    print()

    report = check_image_lookup(resolve_path(cfg, "patient_csv"), dirs, labeled)
    summarize_lookup(report)

    out = ensure_dir(resolve_path(cfg, "results_dir")) / "data_check.csv"
    report.to_csv(out, index=False)
    print(f"\nFull report -> {out}")


if __name__ == "__main__":
    main()
