# Models

This folder is gitignored. It holds:

| File | Source |
|------|--------|
| `best.pt` | Trained YOLO weights (detects `C` and `1` markers). Provide this yourself. |
| `calibration.json` | Written by `scripts/01_calibrate_bands.py` (band position mean/std in cm). |
| `final_model_band{1..6}.pth` | Written by `scripts/03_train_regressors.py`. |

Share weights via GitHub Releases or Git LFS rather than committing them.
