"""Patient condition classifier trained on the six predicted band concentrations.

Evaluated with stratified K-fold CV; out-of-fold predictions, probabilities and
16-d penultimate-layer embeddings are returned for analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data_check import unify_case

FEATURE_COLS = [f"Band_{i}_Prediction" for i in range(1, 7)]


class ConditionClassifier(nn.Module):
    def __init__(self, input_size=6, num_classes=2, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_classification_data(csv_path, feature_cols=FEATURE_COLS, condition_aliases=None):
    """Read ``patient_predictions.csv`` and keep clean, labelled rows.

    Labels that differ only by case (``Negative`` / ``negative``) are always treated as
    one class. ``condition_aliases`` optionally merges *different* labels, e.g.
    ``{"positive": "HIV1"}``. Returns ``(data, X, y, encoder)``.
    """
    df = pd.read_csv(csv_path)
    data = df.loc[df["Status"].eq("OK")].copy()
    data["Condition"] = data["Condition"].astype("string").str.strip()
    if condition_aliases:
        data["Condition"] = data["Condition"].replace(condition_aliases)
    data["Condition"] = unify_case(data["Condition"]).astype("string")
    data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_cols + ["Condition"])
    data = data[data["Condition"] != ""].reset_index(drop=True)

    X = data[feature_cols].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["Condition"].astype(str))
    return data, X, y, encoder


def cross_validate_classifier(X, y, classes, n_splits=5, epochs=200, lr=0.001,
                              weight_decay=1e-4, dropout=0.2, seed=42, device=None,
                              verbose=True):
    """Stratified K-fold CV (preprocessing fit on the training fold only).

    Returns a dict with ``oof_predictions``, ``oof_probabilities``,
    ``oof_embeddings``, ``fold_metrics`` (DataFrame) and ``report`` (str).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(classes)
    class_counts = np.bincount(y)
    if class_counts.min() < n_splits:
        raise ValueError(
            f"{n_splits}-fold stratified CV requires at least {n_splits} samples in every "
            f"class. Counts: {dict(zip(classes, class_counts))}"
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_predictions = np.full(len(y), -1, dtype=int)
    oof_probabilities = np.zeros((len(y), n_classes))
    oof_embeddings = np.full((len(y), 16), np.nan)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        if verbose:
            print(f"\n{'=' * 20} FOLD {fold}/{n_splits} {'=' * 20}")

        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_test = scaler.transform(X[test_idx]).astype(np.float32)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

        torch.manual_seed(seed + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold)

        model = ConditionClassifier(X.shape[1], n_classes, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(X_test_t), dim=1)
            predictions = probabilities.argmax(dim=1).cpu().numpy()
            embeddings = model.network[:-1](X_test_t).cpu().numpy()  # penultimate layer (16-d)

        oof_predictions[test_idx] = predictions
        oof_probabilities[test_idx] = probabilities.cpu().numpy()
        oof_embeddings[test_idx] = embeddings

        fold_results.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, predictions),
            "Macro_F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        })
        if verbose:
            r = fold_results[-1]
            print(f"Accuracy:          {r['Accuracy']:.4f}")
            print(f"Balanced accuracy: {r['Balanced_Accuracy']:.4f}")
            print(f"Macro F1:          {r['Macro_F1']:.4f}")

    fold_df = pd.DataFrame(fold_results)
    report = classification_report(
        y, oof_predictions, labels=np.arange(n_classes), target_names=classes, zero_division=0
    )
    if verbose:
        print("\nPer-fold metrics:")
        print(fold_df.round(4).to_string(index=False))
        print("\nMean and standard deviation across folds:")
        print(fold_df.drop(columns="Fold").agg(["mean", "std"]).round(4))
        print("\nOverall out-of-fold classification report:")
        print(report)

    return {
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "oof_embeddings": oof_embeddings,
        "fold_metrics": fold_df,
        "report": report,
        "confusion_matrix": confusion_matrix(y, oof_predictions, labels=np.arange(n_classes)),
    }


def _id_cols(data):
    return ["Image", "Condition"] + [c for c in ("Image_Dir", "Image_Path") if c in data.columns]


def build_prediction_table(data, y, oof_predictions, oof_probabilities, encoder):
    """Per-image out-of-fold predictions with class probabilities."""
    table = data[_id_cols(data)].copy().reset_index(drop=True)
    table["Actual_Class"] = encoder.inverse_transform(y)
    table["Predicted_Class"] = encoder.inverse_transform(oof_predictions)
    table["Correct"] = table["Actual_Class"] == table["Predicted_Class"]
    for i, name in enumerate(encoder.classes_):
        table[f"Probability_{name}"] = oof_probabilities[:, i]
    return table


def find_misclassified(data, y, oof_predictions, encoder, feature_cols=FEATURE_COLS):
    """Rows the out-of-fold classifier got wrong, with their six predicted concentrations."""
    error_df = data[_id_cols(data)].copy().reset_index(drop=True)
    error_df["Actual_Class"] = encoder.inverse_transform(y)
    error_df["Predicted_Class"] = encoder.inverse_transform(oof_predictions)
    error_df["Correct"] = error_df["Actual_Class"] == error_df["Predicted_Class"]
    for col in feature_cols:
        error_df[col] = data[col].to_numpy()

    misclassified = (
        error_df.loc[~error_df["Correct"]]
        .sort_values(["Actual_Class", "Predicted_Class", "Image"])
        .reset_index(drop=True)
    )
    return error_df, misclassified
