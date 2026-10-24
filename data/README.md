# Data layout

This folder is gitignored. Place your data here (or point `configs/config.yaml` elsewhere).

```
data/
├── calibration/
│   ├── 1/    # images of strips at concentration 1
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
├── patient_samples/        # patient strip images (listed in patient_readings.csv)
├── positive_control/       # ALL images analysed, labelled "positive"
├── negative_control/       # ALL images analysed, labelled "negative"
└── patient_readings.csv    # one row per image in patient_samples/
```

## Which images are analysed

Step 04 analyses two groups of images:

1. **The rows of `patient_readings.csv`**, found by filename in `patient_samples/`.
2. **Every image in `positive_control/` and `negative_control/`.** These do *not* need to be listed in the CSV. Each image becomes a row labelled `positive` or `negative` according to its folder (set by `paths.labeled_image_dirs` in `configs/config.yaml`).

Example: 61 CSV rows + 52 control images = 113 images, all used by the step 05 classifier.

Run `python scripts/check_data.py` to see the full list, where each image was found, and any missing or duplicated filenames. The predictions file records the source folder (`Image_Dir`) and exact file (`Image_Path`) of every image, so same-named files in different folders never get mixed up.

## `patient_readings.csv`

| Column | Required | Description |
|--------|----------|-------------|
| `Image` | yes | Filename of an image in `patient_samples/` (e.g. `IMG_0001.jpg`) |
| `Condition` | for step 05 | Label, e.g. `HIV1`, `Negative` |
| `Notes` | no | Free text. An unlabeled third column is also treated as notes. |

Every image must show the two control markers (`C`) and the signal marker (`1`) for YOLO detection.

### Labels

- Labels that differ only by case or surrounding spaces (`Negative` / `negative`) are **merged automatically** into one class. The first spelling seen wins, and CSV rows come first.
- Different words stay different. The control folders are labelled `positive` and `Negative`; the CSV uses `HIV1` and `Negative`. So `HIV1` and `positive` are separate classes unless you merge them with `classifier.condition_aliases` in the config, e.g. `{positive: HIV1}`.

## Privacy

If images come from real patients, keep them out of version control and make sure you have permission to share anything derived from them.
