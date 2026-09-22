"""Temperature scaling for relation-classification confidence calibration."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Fit one positive scalar temperature on held-out logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(initial_temperature)), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature by minimizing validation negative log-likelihood."""
        x = torch.as_tensor(logits, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )
        criterion = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(x), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())


def softmax_with_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)
