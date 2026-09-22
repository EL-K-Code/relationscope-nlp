"""Metrics shared by the legacy notebook reproductions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def legacy_classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "micro_f1": float(f1_score(labels, predictions, average="micro")),
    }
