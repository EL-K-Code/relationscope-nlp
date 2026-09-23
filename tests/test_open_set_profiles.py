from __future__ import annotations

import pytest
from pydantic import ValidationError

from relationscope.data.schema import EntitySpan, OpenSetPredictRequest
from relationscope.inference.predictor import (
    available_open_set_profiles,
    default_open_set_profile,
    resolve_open_set_threshold,
)

PROFILE_CONFIG: dict[str, object] = {
    "temperature": 1.086841344833374,
    "abstention_threshold": 0.967,
    "default_profile": "balanced",
    "profiles": {
        "high_coverage": {"threshold": 0.924},
        "balanced": {"threshold": 0.967},
        "high_safety": {"threshold": 0.978},
    },
}


def test_profile_thresholds_are_resolved_from_bundle_config() -> None:
    assert available_open_set_profiles(PROFILE_CONFIG) == [
        "high_coverage",
        "balanced",
        "high_safety",
    ]
    assert default_open_set_profile(PROFILE_CONFIG) == "balanced"
    assert resolve_open_set_threshold(PROFILE_CONFIG, "high_coverage") == pytest.approx(
        0.924
    )
    assert resolve_open_set_threshold(PROFILE_CONFIG, "balanced") == pytest.approx(0.967)
    assert resolve_open_set_threshold(PROFILE_CONFIG, "high_safety") == pytest.approx(
        0.978
    )


def test_legacy_bundle_keeps_balanced_profile_compatibility() -> None:
    legacy: dict[str, object] = {
        "temperature": 1.1,
        "abstention_threshold": 0.95,
    }

    assert available_open_set_profiles(legacy) == ["balanced"]
    assert default_open_set_profile(legacy) == "balanced"
    assert resolve_open_set_threshold(legacy, "balanced") == pytest.approx(0.95)

    with pytest.raises(RuntimeError, match="high_safety"):
        resolve_open_set_threshold(legacy, "high_safety")


def test_open_set_request_defaults_to_balanced_profile() -> None:
    request = OpenSetPredictRequest(
        text="Alice works at Acme.",
        head=EntitySpan(text="Alice", start=0, end=5),
        tail=EntitySpan(text="Acme", start=15, end=19),
    )

    assert request.profile == "balanced"


def test_open_set_request_rejects_unknown_profile() -> None:
    with pytest.raises(ValidationError):
        OpenSetPredictRequest(
            text="Alice works at Acme.",
            head=EntitySpan(text="Alice", start=0, end=5),
            tail=EntitySpan(text="Acme", start=15, end=19),
            profile="experimental",  # type: ignore[arg-type]
        )
