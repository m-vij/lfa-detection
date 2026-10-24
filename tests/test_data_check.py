"""Patient CSV loading and multi-folder image lookup (no torch needed)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from lfa.config import image_search_dirs, load_config
from lfa.data_check import (build_patient_table, check_image_lookup, find_all, find_image,
                            load_patient_csv, unify_case)
from lfa.visualization import plot_misclassified


@pytest.fixture
def folders(tmp_path):
    dirs = {}
    for name in ("patient_samples", "positive_control", "negative_control"):
        d = tmp_path / name
        d.mkdir()
        dirs[name] = d
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(dirs["patient_samples"] / "p1.jpg"), img)
    cv2.imwrite(str(dirs["positive_control"] / "pos1.jpg"), img)
    cv2.imwrite(str(dirs["negative_control"] / "neg1.png"), img)
    return dirs


@pytest.fixture
def csv_path(tmp_path):
    # Same shape as the real file: unlabeled third column holds notes.
    path = tmp_path / "patient_readings.csv"
    path.write_text("Image,Condition,\n"
                    "p1.jpg,HIV1,weak bands\n"
                    "pos1.jpg,HIV1,\n"
                    "neg1.png,Negative,\n"
                    "ghost.jpg,HIV1,\n")
    return path


def test_config_lists_control_folders():
    names = [d.name for d in image_search_dirs(load_config())]
    assert names == ["patient_samples", "positive_control", "negative_control"]


def test_find_image_searches_all_folders(folders):
    dirs = list(folders.values())
    assert find_image("p1.jpg", dirs).parent.name == "patient_samples"
    assert find_image("pos1.jpg", dirs).parent.name == "positive_control"
    assert find_image("neg1.png", dirs).parent.name == "negative_control"  # non-.jpg extension
    assert find_image("missing.jpg", dirs) is None
    assert find_image("nan", dirs) is None
    # old single-folder call style still works, and only sees that folder
    assert find_image("pos1.jpg", folders["patient_samples"]) is None


def test_duplicate_filenames_are_reported(folders):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(folders["negative_control"] / "p1.jpg"), img)  # same name in two folders
    matches = find_all("p1.jpg", list(folders.values()))
    assert len(matches) == 2
    assert matches[0].parent.name == "patient_samples"  # folder order decides


def test_load_patient_csv_reads_unlabeled_notes_column(csv_path):
    patients = load_patient_csv(csv_path)
    assert {"Image", "Condition", "Notes"} <= set(patients.columns)
    assert patients.loc[0, "Notes"] == "weak bands"
    assert patients.loc[1, "Notes"] == ""


def test_check_image_lookup_report(folders, csv_path):
    report = check_image_lookup(csv_path, list(folders.values()))
    assert report.set_index("Image")["Found_In"].to_dict() == {
        "p1.jpg": "patient_samples", "pos1.jpg": "positive_control",
        "neg1.png": "negative_control", "ghost.jpg": "MISSING"}
    assert (report["N_Matches"] <= 1).all()


def test_misclassified_gallery_finds_control_images(folders, tmp_path):
    mis = pd.DataFrame({"Image": ["pos1.jpg", "neg1.png", "nowhere.jpg"],
                        "Actual_Class": ["HIV1", "Negative", "HIV1"],
                        "Predicted_Class": ["Negative", "HIV1", "Negative"]})
    out = tmp_path / "gallery.png"
    plot_misclassified(mis, list(folders.values()), save_path=out)
    assert out.is_file()


def _write(folder, name):
    cv2.imwrite(str(folder / name), np.zeros((20, 20, 3), dtype=np.uint8))


def test_labeled_folders_add_every_image_with_folder_label(folders, csv_path):
    _write(folders["positive_control"], "pos2.jpg")
    _write(folders["positive_control"], "pos3.jpg")
    _write(folders["negative_control"], "neg2.jpg")
    labeled = {folders["positive_control"]: "positive", folders["negative_control"]: "negative"}

    table = build_patient_table(csv_path, list(folders.values()), labeled)

    extra = table[table["Source"] != "csv"]
    assert sorted(extra["Image"]) == ["neg2.jpg", "pos2.jpg", "pos3.jpg"]
    # the CSV spells the negative class "Negative", so the folder label "negative" joins it
    assert extra.set_index("Image")["Condition"].to_dict() == {
        "pos2.jpg": "positive", "pos3.jpg": "positive", "neg2.jpg": "Negative"}
    assert (extra["Path"] != "").all()
    assert (table[table["Source"] == "csv"]["Path"] == "").all()


def test_file_listed_in_csv_is_not_double_counted(folders, csv_path):
    # csv_path lists pos1.jpg and neg1.png, which also sit in the labeled folders.
    labeled = {folders["positive_control"]: "positive", folders["negative_control"]: "negative"}
    table = build_patient_table(csv_path, list(folders.values()), labeled)
    assert table["Image"].tolist().count("pos1.jpg") == 1
    assert table["Image"].tolist().count("neg1.png") == 1
    assert len(table) == 4  # just the 4 CSV rows; nothing extra


def test_same_filename_in_two_folders_stays_separate(folders, csv_path):
    # A *different* image named p1.jpg in positive_control must not be swallowed by the
    # CSV's p1.jpg (which lives in patient_samples).
    _write(folders["positive_control"], "p1.jpg")
    labeled = {folders["positive_control"]: "positive"}
    table = build_patient_table(csv_path, list(folders.values()), labeled)

    p1 = table[table["Image"] == "p1.jpg"]
    assert len(p1) == 2
    assert set(p1["Source"]) == {"csv", "positive_control"}
    assert p1[p1["Source"] == "positive_control"]["Path"].iloc[0].endswith("positive_control/p1.jpg")

    report = check_image_lookup(csv_path, list(folders.values()), labeled)
    assert report[report["Image"] == "p1.jpg"]["Duplicate_Name"].all()


def test_check_image_lookup_counts_labeled_rows(folders, csv_path):
    _write(folders["negative_control"], "neg2.jpg")
    labeled = {folders["negative_control"]: "negative"}
    report = check_image_lookup(csv_path, list(folders.values()), labeled)
    row = report[report["Image"] == "neg2.jpg"].iloc[0]
    assert row["Condition"] == "Negative" and row["Found_In"] == "negative_control"


def test_labeled_dirs_missing_folder_is_skipped(folders, csv_path, tmp_path):
    table = build_patient_table(csv_path, list(folders.values()), {tmp_path / "nope": "x"})
    assert len(table) == 4


def test_unify_case_merges_case_variants_only():
    out = unify_case(["Negative", "negative", "NEGATIVE ", "HIV1", "positive", "", None])
    assert out.tolist()[:3] == ["Negative", "Negative", "Negative"]
    assert out.tolist()[3:5] == ["HIV1", "positive"]       # different words stay different
    assert out.tolist()[5] == "" and pd.isna(out.tolist()[6])  # missing stays missing (None or NaN, depends on pandas)


def test_negative_and_Negative_become_one_class_in_the_table(folders, csv_path):
    _write(folders["negative_control"], "neg2.jpg")
    labeled = {folders["negative_control"]: "negative"}
    table = build_patient_table(csv_path, list(folders.values()), labeled)
    assert set(table["Condition"]) == {"HIV1", "Negative"}


def test_classifier_loader_treats_case_variants_as_one_class(tmp_path):
    pytest.importorskip("torch")  # classification.py imports torch
    from lfa.classification import FEATURE_COLS, load_classification_data

    rows = []
    for i, label in enumerate(["Negative", "negative", "NEGATIVE", "HIV1", "positive"]):
        rows.append({"Image": f"x{i}.jpg", "Condition": label, "Status": "OK",
                     **{c: 0.1 * (i + 1) for c in FEATURE_COLS}})
    path = tmp_path / "pred.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    _, _, y, enc = load_classification_data(path)
    assert sorted(enc.classes_) == ["HIV1", "Negative", "positive"]
    assert np.bincount(y).tolist().count(3) == 1  # three negative spellings -> one class of 3
