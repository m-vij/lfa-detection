"""Patient image inference (no retraining).

``PatientPipeline`` bundles the YOLO detector, calibrated band positions and
the six trained regressors, and turns a patient image into per-band intensities,
ratios and predicted concentrations.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibration import NUM_BANDS, load_calibration
from .data_check import as_dirs, build_patient_table, find_image
from .detection import load_yolo, locate_strip
from .models import band_model_path, load_band_model, predict_band
from .signal_processing import sample_signal_using_distance
from .visualization import plot_patient_diagnostic

log = logging.getLogger(__name__)


class PatientPipeline:
    def __init__(self, yolo_weights, calibration_json, models_dir, cfg,
                 plot_dir=None, device="cpu"):
        self.cfg = cfg
        self.plot_dir = Path(plot_dir) if plot_dir else None
        if self.plot_dir:
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        if not Path(yolo_weights).is_file():
            raise FileNotFoundError(f"YOLO model not found: {yolo_weights}")
        self.yolo, self.class_names = load_yolo(yolo_weights)
        self.band_means, self.band_stds = load_calibration(calibration_json)

        self.band_models = {}
        for band in range(1, NUM_BANDS + 1):
            path = band_model_path(models_dir, band)
            self.band_models[band] = load_band_model(path, device)
            log.info("Loaded band %d: %s", band, path.name)

    # -- image lookup (delegates to lfa.data_check) ------------------------
    find_image = staticmethod(find_image)

    # -- single image -------------------------------------------------------
    def process_image(self, image_path, image_name, condition="", notes=""):
        result = {"Image": str(image_name), "Condition": condition, "Notes": notes,
                  "Image_Dir": Path(image_path).parent.name,  # which folder it was found in
                  "Image_Path": str(image_path),
                  "Status": "ERROR", "Error": ""}
        sig = self.cfg["signal"]

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            detections = self.yolo(str(image_path))
            if len(detections) == 0:
                raise ValueError("YOLO returned no detections.")

            midpoint, slope, distance, target = locate_strip(detections[0], self.class_names)

            _, intensities = sample_signal_using_distance(
                img, midpoint, target, slope, distance,
                self.band_means, self.band_stds,
                track_width=sig["track_width"], tolerance=sig["tolerance"],
                rw_distance=self.cfg["geometry"]["real_world_distance_cm"],
                control_prominence=sig["control_prominence"],
                peak_width=sig["peak_width"],
                savgol_window=sig["savgol_window"],
                savgol_polyorder=sig["savgol_polyorder"],
                baseline_sigma=sig["baseline_sigma"],
            )
            if len(intensities) < 7:
                raise ValueError(f"Expected 7 peak intensities; got {len(intensities)}")

            control = float(intensities[0])
            result["Control_Intensity"] = control

            for band in range(1, NUM_BANDS + 1):
                signal = float(intensities[band])
                result[f"Band_{band}_Intensity"] = signal
                result[f"Band_{band}_Ratio"] = signal / control if control != 0 else np.nan

                # Models are trained on raw [Control, Signal], not on the ratio alone.
                if not np.isfinite(control) or not np.isfinite(signal):
                    raise ValueError(f"Non-finite intensity for band {band}")
                result[f"Band_{band}_Prediction"] = predict_band(
                    control, signal, self.band_models[band]
                )

            if self.plot_dir:
                plot_path = self.plot_dir / (
                    f"{Path(image_path).parent.name}_{Path(image_name).stem}_diagnostic.png"
                )
                plot_patient_diagnostic(img, image_name, intensities, plot_path)
                result["Diagnostic_Plot"] = str(plot_path)

            result["Status"] = "OK"

        except Exception as error:  # noqa: BLE001 - one bad image must not stop the batch
            result["Error"] = f"{type(error).__name__}: {error}"

        return result

    # -- whole CSV ----------------------------------------------------------
    def run(self, patient_csv, image_dirs, labeled_dirs=None):
        """Process every CSV image plus every image in ``labeled_dirs``.

        ``image_dirs``  -- folder(s) searched for the filenames in the CSV.
        ``labeled_dirs`` -- ``{folder: label}``; each image in each folder becomes a row
        labelled ``label`` (e.g. ``{positive_control: "positive"}``).
        """
        for d in as_dirs(image_dirs) + [Path(p) for p in (labeled_dirs or {})]:
            if not d.is_dir():
                log.warning("Image folder not found: %s", d)
        patients = build_patient_table(patient_csv, image_dirs, labeled_dirs)

        all_results = []
        for index, patient in patients.iterrows():
            image_name = patient["Image"]
            condition = patient.get("Condition", "")
            notes = patient.get("Notes", "")

            log.info("[%d/%d] Processing %s", index + 1, len(patients), image_name)
            if patient["Path"]:  # folder row: exact file known
                image_path = Path(patient["Path"])
            else:
                image_path = self.find_image(image_name, image_dirs)

            if image_path is None:
                result = {"Image": image_name, "Condition": condition, "Notes": notes,
                          "Image_Dir": "", "Image_Path": "", "Status": "ERROR",
                          "Error": f"Image not found: {image_name}"}
            else:
                result = self.process_image(image_path, image_name, condition, notes)

            if result["Status"] == "ERROR":
                log.warning("  %s", result["Error"])
            all_results.append(result)

        return pd.DataFrame(all_results)
