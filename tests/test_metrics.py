import numpy as np
import pytest

from relationscope.training.metrics import classification_metrics, expected_calibration_error


def test_metrics_are_perfect_for_perfect_predictions() -> None:
    probabilities = np.array([[0.99, 0.01], [0.02, 0.98]])
    labels = np.array([0, 1])

    metrics = classification_metrics(probabilities, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert 0.0 <= metrics["ece_15"] <= 0.03


def test_ece_rejects_wrong_shape() -> None:
    probabilities = np.array([0.9, 0.1])
    labels = np.array([0])

    with pytest.raises(ValueError, match="n_samples"):
        expected_calibration_error(probabilities, labels)
