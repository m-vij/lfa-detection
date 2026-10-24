"""Patient table construction and image lookup (no heavy dependencies).

The set of images to analyse has two sources:

1. Rows in ``patient_readings.csv`` (image filename + condition label + notes).
   These are located by filename in ``patient_samples/`` (and other search folders).
2. *Labeled folders*, e.g. ``positive_control/`` and ``negative_control/``.
   Every image in such a folder becomes its own row, labelled from the folder
   (``positive`` / ``negative``). These rows carry their exact file path, so two
   folders may safely contain files with the same name.

``build_patient_table`` merges both sources; ``check_image_lookup`` reports what
it found so problems are visible before running the model.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from .detection import IMAGE_EXTENSIONS

log = logging.getLogger(__name__)


def as_dirs(image_dirs):
    """Accept one folder or an ordered iterable of folders; return a list of Paths."""
    if isinstance(image_dirs, (str, Path)):
        return [Path(image_dirs)]
    return [Path(d) for d in image_dirs]


def list_images(directory):
    """Sorted image paths in ``directory`` (skips hidden files / non-images)."""
    directory = Path(directory)
    return [
        str(directory / f)
        for f in sorted(os.listdir(directory))
        if not f.startswith(".") and Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_patient_csv(patient_csv):
    """Read ``patient_readings.csv`` and normalise its columns.

    Requires ``Image``. Adds an empty ``Condition`` if absent. An unlabeled
    third column (pandas names it ``Unnamed: 2``) is treated as ``Notes``.
    """
    patients = pd.read_csv(patient_csv)

    if "Image" not in patients.columns:
        raise ValueError(f"Expected 'Image' column. Found {patients.columns.tolist()}")
    if "Condition" not in patients.columns:
        patients["Condition"] = ""
    if "Unnamed: 2" in patients.columns:
        patients["Notes"] = patients["Unnamed: 2"].fillna("")
    elif "Notes" not in patients.columns:
        patients["Notes"] = ""

    patients["Image"] = patients["Image"].astype(str).str.strip()
    return patients


def find_all(image_name, image_dirs):
    """Every file matching ``image_name``: direct hits first (folder order), then subfolders."""
    image_name = str(image_name).strip()
    if not image_name or image_name.lower() == "nan":
        return []

    dirs = as_dirs(image_dirs)
    candidate = Path(image_name)
    if candidate.is_absolute():
        return [candidate] if candidate.is_file() else []

    found = []
    for d in dirs:
        direct = d / candidate
        if direct.is_file():
            found.append(direct)
    for d in dirs:
        if d.is_dir():
            for p in d.rglob(candidate.name):
                if p.is_file() and p not in found:
                    found.append(p)
    return found


def find_image(image_name, image_dirs):
    """First match for ``image_name`` (see ``find_all``), or ``None``."""
    matches = find_all(image_name, image_dirs)
    if len(matches) > 1:
        log.warning("Multiple files named %s; using %s", Path(str(image_name)).name, matches[0])
    return matches[0] if matches else None


def unify_case(labels):
    """Treat labels that differ only by case (or surrounding spaces) as the same label.

    The first spelling encountered is used for every variant, e.g. ``Negative`` and
    ``negative`` both become whichever appeared first. Other text is unchanged, so
    ``HIV1`` and ``positive`` stay different. Missing values and ``""`` are preserved.
    """
    series = pd.Series(labels)
    canonical = {}
    out = []
    for v in series:
        if isinstance(v, str):
            v = v.strip()
            key = v.casefold()
            if key:
                canonical.setdefault(key, v)
                v = canonical[key]
        out.append(v)
    return pd.Series(out, index=series.index)


def build_patient_table(patient_csv, image_dirs, labeled_dirs=None):
    """CSV rows plus every image found in the labeled folders.

    Parameters
    ----------
    patient_csv : path to ``patient_readings.csv``
    image_dirs : folders searched (in order) for the filenames listed in the CSV
    labeled_dirs : ``{folder: label}`` -- every image in ``folder`` is added as a row
        with ``Condition = label``. If the CSV already lists that exact file, the CSV
        row is kept and the folder copy is skipped (no double counting).

    Condition labels that differ only by case are unified (see ``unify_case``).

    Returns a DataFrame with ``Image``, ``Condition``, ``Notes``, ``Path`` (exact file
    for folder rows, empty for CSV rows) and ``Source`` (``csv`` or the folder name).
    """
    patients = load_patient_csv(patient_csv)
    patients["Path"] = ""
    patients["Source"] = "csv"
    listed = set(patients["Image"])

    rows = []
    for folder, label in (labeled_dirs or {}).items():
        folder = Path(folder)
        if not folder.is_dir():
            log.warning("Labeled image folder not found: %s", folder)
            continue
        for path in list_images(folder):
            name = Path(path).name
            if name in listed:
                hit = find_image(name, image_dirs)
                if hit is not None and Path(hit).resolve() == Path(path).resolve():
                    log.info("%s is already listed in the CSV; skipping folder copy", name)
                    continue
            rows.append({"Image": name, "Condition": label, "Notes": "",
                         "Path": path, "Source": folder.name})

    if rows:
        patients = pd.concat([patients, pd.DataFrame(rows)], ignore_index=True)
    # CSV rows come first, so the CSV's spelling (e.g. "Negative") wins over the folder
    # label ("negative"); the two are then one class everywhere downstream.
    patients["Condition"] = unify_case(patients["Condition"])
    return patients


def check_image_lookup(patient_csv, image_dirs, labeled_dirs=None):
    """Report where every image to be analysed is found.

    Returns a DataFrame with ``Image``, ``Condition``, ``Notes``, ``Source``,
    ``Found_In`` (folder of the first match, or ``MISSING``), ``N_Matches``,
    ``Duplicate_Name`` (same filename used by more than one row) and ``All_Matches``.
    """
    table = build_patient_table(patient_csv, image_dirs, labeled_dirs)
    dup_names = table["Image"].duplicated(keep=False)

    rows = []
    for i, row in table.iterrows():
        matches = [Path(row["Path"])] if row["Path"] else find_all(row["Image"], image_dirs)
        rows.append({
            "Image": row["Image"],
            "Condition": row["Condition"],
            "Notes": row["Notes"],
            "Source": row["Source"],
            "Found_In": matches[0].parent.name if matches else "MISSING",
            "N_Matches": len(matches),
            "Duplicate_Name": bool(dup_names.iloc[i]),
            "All_Matches": "; ".join(str(m) for m in matches),
        })
    return pd.DataFrame(rows)


def summarize_lookup(report):
    """Print a short summary of ``check_image_lookup`` output."""
    print(f"Total images to analyse: {len(report)}")
    print("\nRows by source (csv = listed in patient_readings.csv):")
    print(report["Source"].value_counts().to_string())
    print("\nCondition x folder:")
    print(pd.crosstab(report["Condition"], report["Found_In"]).to_string())

    missing = report[report["Found_In"] == "MISSING"]
    dupes = report[report["N_Matches"] > 1]
    same_name = report[report["Duplicate_Name"]]
    print(f"\nMissing: {len(missing)} | CSV filename found in more than one folder: {len(dupes)}"
          f" | Same filename used by several rows: {len(same_name)}")
    if len(missing):
        print("  missing ->", ", ".join(missing["Image"].tolist()))
    if len(dupes):
        print("  in several folders ->", ", ".join(dupes["Image"].tolist()))
    if len(same_name):
        print("  same filename in several rows (kept apart by exact path) ->",
              ", ".join(sorted(set(same_name["Image"]))))
    return missing, dupes
