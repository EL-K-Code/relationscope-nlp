"""Champion/challenger promotion gates for RelationScope v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from relationscope.data.v3_schema import PromotionGateCheck, PromotionGateResult


def _get(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


class PromotionGate:
    """Compare a challenger evaluation report against the promoted champion."""

    def __init__(
        self,
        *,
        known_macro_f1_tolerance: float = 0.005,
        unknown_f1_tolerance: float = 0.01,
        auroc_tolerance: float = 0.01,
        unknown_rejection_tolerance: float = 0.02,
        accepted_known_accuracy_tolerance: float = 0.005,
        max_latency_multiplier: float = 1.25,
    ) -> None:
        self.known_macro_f1_tolerance = known_macro_f1_tolerance
        self.unknown_f1_tolerance = unknown_f1_tolerance
        self.auroc_tolerance = auroc_tolerance
        self.unknown_rejection_tolerance = unknown_rejection_tolerance
        self.accepted_known_accuracy_tolerance = accepted_known_accuracy_tolerance
        self.max_latency_multiplier = max_latency_multiplier

    def compare(
        self,
        champion: dict[str, Any],
        challenger: dict[str, Any],
    ) -> PromotionGateResult:
        checks: list[PromotionGateCheck] = []
        warnings: list[str] = []

        champion_model = str(_get(champion, "model", "model_version"))
        challenger_model = str(_get(challenger, "model", "model_version"))

        champion_protocol = str(_get(champion, "protocol", "name"))
        challenger_protocol = str(_get(challenger, "protocol", "name"))
        protocol_passed = champion_protocol == challenger_protocol
        checks.append(
            PromotionGateCheck(
                name="evaluation_protocol",
                champion=champion_protocol,
                challenger=challenger_protocol,
                passed=protocol_passed,
                detail=(
                    "Evaluation protocols match."
                    if protocol_passed
                    else "Promotion requires the exact same frozen evaluation protocol."
                ),
            )
        )

        numeric_specs = [
            (
                "known_macro_f1",
                ("known_test", "macro_f1"),
                self.known_macro_f1_tolerance,
                "higher",
            ),
            (
                "open_set_unknown_f1",
                ("final_unseen_test_balanced", "unknown_f1"),
                self.unknown_f1_tolerance,
                "higher",
            ),
            (
                "open_set_auroc",
                ("final_unseen_test_balanced", "auroc"),
                self.auroc_tolerance,
                "higher",
            ),
            (
                "unknown_rejection",
                ("final_unseen_test_balanced", "unknown_rejection"),
                self.unknown_rejection_tolerance,
                "higher",
            ),
            (
                "accepted_known_accuracy",
                ("profiles", "balanced", "accepted_known_accuracy"),
                self.accepted_known_accuracy_tolerance,
                "higher",
            ),
        ]

        for name, path, tolerance, direction in numeric_specs:
            champion_value = float(_get(champion, *path))
            challenger_value = float(_get(challenger, *path))
            if direction == "higher":
                passed = challenger_value >= champion_value - tolerance
                detail = (
                    f"Challenger must be >= champion - {tolerance:.4f}."
                )
            else:
                passed = challenger_value <= champion_value + tolerance
                detail = (
                    f"Challenger must be <= champion + {tolerance:.4f}."
                )
            checks.append(
                PromotionGateCheck(
                    name=name,
                    champion=champion_value,
                    challenger=challenger_value,
                    tolerance=tolerance,
                    passed=passed,
                    detail=detail,
                )
            )

        try:
            champion_latency = float(
                _get(
                    champion,
                    "latency",
                    "cloud_run_cpu_warm_ms",
                    "application_median",
                )
            )
            challenger_latency = float(
                _get(
                    challenger,
                    "latency",
                    "cloud_run_cpu_warm_ms",
                    "application_median",
                )
            )
            max_latency = champion_latency * self.max_latency_multiplier
            checks.append(
                PromotionGateCheck(
                    name="warm_cpu_latency",
                    champion=champion_latency,
                    challenger=challenger_latency,
                    tolerance=max_latency - champion_latency,
                    passed=challenger_latency <= max_latency,
                    detail=(
                        "Challenger warm CPU median must stay within "
                        f"{self.max_latency_multiplier:.2f}x champion latency."
                    ),
                )
            )
        except KeyError:
            warnings.append(
                "Latency gate skipped because one report lacks the warm Cloud Run "
                "application median."
            )

        return PromotionGateResult(
            champion_model=champion_model,
            challenger_model=challenger_model,
            passed=all(check.passed for check in checks),
            checks=checks,
            warnings=warnings,
        )


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation report must contain a JSON object: {path}")
    return payload
