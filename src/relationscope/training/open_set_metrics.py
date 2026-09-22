"""Metrics for unseen-relation detection and selective prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def max_softmax_score(probabilities: np.ndarray) -> np.ndarray:
    """Return maximum class probability for each example."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [n_samples, n_classes]")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one example")
    return probabilities.max(axis=1)


def _unknown_targets(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(unknown_scores), dtype=np.int64),
        ]
    )
    unknown_score = np.concatenate([1.0 - known_scores, 1.0 - unknown_scores])
    return y_true, unknown_score


def open_set_metrics(
    known_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Evaluate unknown detection using low maximum-softmax probability."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    known_scores = max_softmax_score(known_probabilities)
    unknown_scores = max_softmax_score(unknown_probabilities)
    y_true, unknown_score = _unknown_targets(known_scores, unknown_scores)

    predicted_unknown = np.concatenate(
        [
            known_scores < threshold,
            unknown_scores < threshold,
        ]
    ).astype(np.int64)

    fpr, tpr, _ = roc_curve(y_true, unknown_score)
    candidates = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[candidates[0]]) if len(candidates) else 1.0

    return {
        "ood_auroc": float(roc_auc_score(y_true, unknown_score)),
        "ood_auprc": float(average_precision_score(y_true, unknown_score)),
        "ood_fpr95": fpr95,
        "unknown_f1": float(f1_score(y_true, predicted_unknown)),
        "known_acceptance_rate": float(np.mean(known_scores >= threshold)),
        "unknown_rejection_rate": float(np.mean(unknown_scores < threshold)),
        "threshold": float(threshold),
    }


def tune_abstention_threshold(
    known_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    *,
    grid_size: int = 1001,
) -> tuple[float, dict[str, float]]:
    """Choose a threshold by maximizing unknown-vs-known F1 on calibration data."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    known_scores = max_softmax_score(known_probabilities)
    unknown_scores = max_softmax_score(unknown_probabilities)
    y_true, _ = _unknown_targets(known_scores, unknown_scores)

    best_threshold = 0.0
    best_f1 = -1.0

    for threshold in np.linspace(0.0, 1.0, grid_size):
        predicted_unknown = np.concatenate(
            [
                known_scores < threshold,
                unknown_scores < threshold,
            ]
        ).astype(np.int64)
        score = float(f1_score(y_true, predicted_unknown))

        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = score

    return best_threshold, open_set_metrics(
        known_probabilities,
        unknown_probabilities,
        threshold=best_threshold,
    )


def risk_coverage_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, list[float]]:
    """Return selective-classification risk as uncertain cases are rejected."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if len(labels) == 0:
        raise ValueError("probabilities and labels must not be empty")

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    order = np.argsort(-confidence)
    ordered_correct = correct[order]

    cumulative_accuracy = np.cumsum(ordered_correct) / np.arange(
        1,
        len(ordered_correct) + 1,
    )
    coverage = np.arange(1, len(ordered_correct) + 1) / len(ordered_correct)
    risk = 1.0 - cumulative_accuracy

    return {
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }
