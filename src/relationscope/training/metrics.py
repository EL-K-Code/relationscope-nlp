"""Metrics for relation classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Compute top-label Expected Calibration Error."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [n_samples, n_classes]")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if len(labels) == 0:
        return 0.0

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        gap = abs(correctness[mask].mean() - confidence[mask].mean())
        ece += float(mask.mean()) * float(gap)
    return float(ece)


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, probabilities.shape[1])
    topk = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, topk, strict=True)]))


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "micro_f1": float(f1_score(labels, predictions, average="micro")),
        "top3_accuracy": top_k_accuracy(probabilities, labels, 3),
        "ece_15": expected_calibration_error(probabilities, labels, bins=15),
    }
