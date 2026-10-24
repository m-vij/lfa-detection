"""All plotting code lives here so the processing modules stay headless-friendly.

Every function accepts ``save_path`` (write a PNG) and/or ``show`` (display in
a notebook). Figures are always closed afterwards unless ``show`` is True.
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .data_check import find_image


def _finish(fig, save_path=None, show=False, dpi=150):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def plot_first_pass(img, x_path, y_path, midpoint, slope, final_signal, peaks,
                    save_path=None, show=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.plot(x_path, y_path, color="lime", lw=2, label="Path (with tolerance)")
    ax1.scatter(*midpoint, color="magenta", s=100, label="Midpoint C (Ref)", zorder=5)
    ax1.set_title(f"Sampling Path following Slope {slope:.3f}")
    ax1.legend()

    ax2.plot(final_signal, color="blue", lw=2, label="Cleaned Signal")
    ax2.fill_between(range(len(final_signal)), final_signal, alpha=0.2, color="blue")
    if len(peaks) > 0:
        ax2.plot(peaks, final_signal[peaks], "x", color="red", markersize=12, mew=2,
                 label="Detected Peaks")
        for p in peaks:
            ax2.text(p, final_signal[p], f"{final_signal[p]:.1f}", color="red", weight="bold")
    ax2.set_title("Quantified Intensity with Peak Detection")
    ax2.set_ylabel("Intensity (AU)")
    ax2.set_xlabel("Pixels along path (including start/end buffer)")
    ax2.legend()

    _finish(fig, save_path, show)


def plot_band_extraction(img, x_path, y_path, midpoint, slope, final_signal,
                         peak_locations, peak_intensities, windows,
                         save_path=None, show=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.plot(x_path, y_path, color="lime", lw=2, label="Sampling Path")
    ax1.scatter(*midpoint, color="magenta", s=100, label="Reference Midpoint", zorder=5)
    ax1.set_title(f"Sampling Path (Slope {slope:.3f})")
    ax1.legend()

    ax2.plot(final_signal, color="blue", lw=2, label="Processed Signal")
    ax2.plot(peak_locations[0], peak_intensities[0], "rx", markersize=12, mew=2,
             label="Control Peak")
    ax2.annotate(f"Control: {peak_intensities[0]:.1f}",
                 (peak_locations[0], peak_intensities[0]))

    last = len(final_signal) - 1
    for i, (lo, hi) in enumerate(windows, start=1):
        ax2.axvspan(max(0, lo), min(last, hi), alpha=0.15)
        ax2.plot(peak_locations[i], peak_intensities[i], "x", markersize=10)
        ax2.annotate(f"Band {i}: {peak_intensities[i]:.1f}",
                     (peak_locations[i], peak_intensities[i]), fontsize=9)

    ax2.set_title("Control-Referenced Band Intensities")
    ax2.set_ylabel("Intensity (AU)")
    ax2.set_xlabel("Pixels along sampling path")
    ax2.legend()

    _finish(fig, save_path, show)


def plot_patient_diagnostic(img, name, peak_intensities, save_path):
    """Per-patient diagnostic: image next to a bar chart of control + band intensities."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(str(name))
    axes[0].axis("off")

    axes[1].bar(["Control"] + [f"Band {i}" for i in range(1, 7)], peak_intensities[:7])
    axes[1].set_ylabel("Processed intensity (AU)")
    axes[1].tick_params(axis="x", rotation=45)

    _finish(fig, save_path, show=False)


# ---------------------------------------------------------------------------
# Regressor diagnostics (step 03)
# ---------------------------------------------------------------------------
def plot_regression_diagnostics(target, ratio, true, pred, out_dir=None, prefix="", show=False):
    """Three plots: concentration vs ratio with NN fit, predicted vs actual, residuals."""
    out_dir = Path(out_dir) if out_dir else None
    sep = f"{prefix}_" if prefix else ""
    idx = np.argsort(pred)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(target, ratio, alpha=0.6, label="Observed")
    plt.plot(pred[idx], np.asarray(ratio)[idx], linewidth=3, label="NN Fit (Final Model)")
    plt.xlabel("Concentration")
    plt.ylabel("Signal / Control")
    plt.title(f"Concentration vs Ratio {prefix}".strip())
    plt.legend()
    plt.grid()
    _finish(fig, out_dir / f"{sep}concentration_vs_ratio.png" if out_dir else None, show)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.7)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title(f"Predicted vs Actual {prefix}".strip())
    plt.grid()
    _finish(fig, out_dir / f"{sep}predicted_vs_actual.png" if out_dir else None, show)

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(true, pred - true, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("True Concentration")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot {prefix}".strip())
    plt.grid()
    _finish(fig, out_dir / f"{sep}residuals.png" if out_dir else None, show)


# ---------------------------------------------------------------------------
# Patient analysis (steps 04-05)
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(successful, prediction_cols, save_path=None, show=True):
    import seaborn as sns

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(successful[prediction_cols].corr(), annot=True, cmap="coolwarm",
                vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Between Band Predictions")
    _finish(fig, save_path, show)


def plot_violin_by_condition(successful, prediction_cols, save_path=None, show=True):
    """Violin plot of the six band predictions for each patient condition."""
    import seaborn as sns

    plot_data = successful.melt(
        id_vars=["Condition"], value_vars=prediction_cols,
        var_name="Band", value_name="Prediction",
    )
    plot_data["Band"] = plot_data["Band"].str.replace("_Prediction", "", regex=False)

    fig = plt.figure(figsize=(14, 7))
    sns.violinplot(data=plot_data, x="Condition", y="Prediction", hue="Band",
                   inner="quartile", cut=0, density_norm="width")
    plt.title("Distribution of Band Predictions by Patient Condition")
    plt.xlabel("Patient Condition")
    plt.ylabel("Predicted Value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Band", bbox_to_anchor=(1.02, 1), loc="upper left")
    _finish(fig, save_path, show)


def plot_confusion_matrix(cm, classes, save_path=None, show=False):
    import seaborn as sns

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Condition")
    plt.ylabel("Actual Condition")
    plt.title("5-Fold CV: Out-of-Fold Confusion Matrix")
    _finish(fig, save_path, show, dpi=300)


def plot_umap_embeddings(embeddings, labels, classes, save_path=None, show=False,
                         random_state=42):
    """UMAP of the held-out 16-d classifier embeddings (requires ``umap-learn``)."""
    import umap  # optional dependency

    coords = umap.UMAP(n_components=2, random_state=random_state).fit_transform(embeddings)
    fig = plt.figure(figsize=(8, 6))
    for i, name in enumerate(classes):
        m = labels == i
        plt.scatter(coords[m, 0], coords[m, 1], label=name, alpha=0.8)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of Held-Out Classifier Embeddings (5-Fold CV)")
    plt.legend()
    _finish(fig, save_path, show)


def plot_misclassified(misclassified, image_dirs, save_path=None, show=False, ncols=3):
    """Grid of every image misclassified in out-of-fold predictions.

    ``image_dirs`` is a folder or an ordered list of folders to search.
    """
    if misclassified.empty:
        print("No misclassified images in the out-of-fold predictions.")
        return

    image_dirs = [Path(image_dirs)] if isinstance(image_dirs, (str, Path)) else [Path(d) for d in image_dirs]
    n_images = len(misclassified)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for ax, (_, row) in zip(axes.flat, misclassified.iterrows()):
        filename = str(row["Image"]).strip()
        exact = row.get("Image_Path", "")
        if isinstance(exact, str) and exact and Path(exact).is_file():
            path = Path(exact)  # exact file (keeps same-named images in different folders apart)
        else:
            path = find_image(filename, image_dirs)
        image = cv2.imread(str(path)) if path is not None else None

        if image is None:
            msg = "Could not read image" if path is not None else f"Image not found\n{filename}"
            ax.text(0.5, 0.5, msg, ha="center", va="center")
            ax.axis("off")
            continue

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{filename}\nActual: {row['Actual_Class']}\nPredicted: {row['Predicted_Class']}",
                     fontsize=10, color="red")
        ax.axis("off")

    for ax in axes.flat[n_images:]:
        ax.axis("off")

    plt.suptitle("Misclassified Images — 5-Fold Cross-Validation")
    _finish(fig, save_path, show)
