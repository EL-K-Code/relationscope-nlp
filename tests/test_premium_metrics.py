import numpy as np

from relationscope.training.calibration import (
    TemperatureScaler,
    softmax_with_temperature,
)
from relationscope.training.open_set_metrics import (
    open_set_metrics,
    tune_abstention_threshold,
)


def test_temperature_scaling_returns_positive_temperature() -> None:
    logits = np.array([[5.0, 0.1], [0.2, 4.0], [2.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 1, 1])

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels, max_iter=20)

    assert temperature > 0.0
    probabilities = softmax_with_temperature(logits, temperature)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_open_set_threshold_separates_easy_known_and_unknown() -> None:
    known = np.array([[0.98, 0.02], [0.95, 0.05], [0.97, 0.03]])
    unknown = np.array([[0.52, 0.48], [0.55, 0.45], [0.51, 0.49]])

    threshold, calibration = tune_abstention_threshold(
        known,
        unknown,
        grid_size=101,
    )
    metrics = open_set_metrics(known, unknown, threshold=threshold)

    assert calibration["unknown_f1"] == 1.0
    assert metrics["known_acceptance_rate"] == 1.0
    assert metrics["unknown_rejection_rate"] == 1.0
