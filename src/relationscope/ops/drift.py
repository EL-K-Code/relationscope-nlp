"""Operational drift detection for RelationScope v3."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from relationscope.data.v3_schema import DriftMetric, DriftSummaryResponse
from relationscope.ops.store import OperationsStore


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _rate(events: list[dict[str, Any]], decision: str) -> float:
    if not events:
        return 0.0
    return sum(event.get("decision") == decision for event in events) / len(events)


def _distribution(events: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(event.get("relation_id") or "UNKNOWN") for event in events)


def _js_divergence(
    first: Counter[str],
    second: Counter[str],
) -> float:
    labels = set(first) | set(second)
    if not labels:
        return 0.0

    total_first = sum(first.values())
    total_second = sum(second.values())
    if total_first == 0 or total_second == 0:
        return 0.0

    p = {label: first[label] / total_first for label in labels}
    q = {label: second[label] / total_second for label in labels}
    m = {label: (p[label] + q[label]) / 2.0 for label in labels}

    def kl(left: dict[str, float], middle: dict[str, float]) -> float:
        value = 0.0
        for label, probability in left.items():
            if probability <= 0.0:
                continue
            value += probability * math.log2(probability / middle[label])
        return value

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class DriftService:
    """Compare the newest telemetry window against the immediately preceding one."""

    def __init__(
        self,
        store: OperationsStore,
        *,
        min_events_per_window: int = 20,
        unknown_rate_delta_threshold: float = 0.15,
        confidence_drop_threshold: float = 0.10,
        text_length_relative_threshold: float = 0.30,
        relation_js_threshold: float = 0.20,
    ) -> None:
        self.store = store
        self.min_events_per_window = min_events_per_window
        self.unknown_rate_delta_threshold = unknown_rate_delta_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.text_length_relative_threshold = text_length_relative_threshold
        self.relation_js_threshold = relation_js_threshold

    def summary(self, *, limit: int = 400) -> DriftSummaryResponse:
        events = [
            event
            for event in self.store.list_json("telemetry", limit=limit)
            if event.get("event_type") == "prediction"
        ]

        if len(events) < self.min_events_per_window * 2:
            return DriftSummaryResponse(
                status="insufficient_data",
                baseline_events=0,
                current_events=len(events),
                relation_js_divergence=0.0,
                metrics=[],
                message=(
                    "Not enough prediction telemetry to compare two drift windows. "
                    f"Need at least {self.min_events_per_window * 2} events."
                ),
            )

        window_size = min(len(events) // 2, limit // 2)
        current = events[:window_size]
        baseline = events[window_size : window_size * 2]

        baseline_unknown = _rate(baseline, "UNKNOWN_RELATION")
        current_unknown = _rate(current, "UNKNOWN_RELATION")
        unknown_delta = current_unknown - baseline_unknown

        baseline_confidence = _mean(
            [float(event.get("confidence", 0.0)) for event in baseline]
        )
        current_confidence = _mean(
            [float(event.get("confidence", 0.0)) for event in current]
        )
        confidence_delta = current_confidence - baseline_confidence

        baseline_length = _mean(
            [float(event.get("text_length", 0.0)) for event in baseline]
        )
        current_length = _mean(
            [float(event.get("text_length", 0.0)) for event in current]
        )
        length_relative_delta = (
            (current_length - baseline_length) / baseline_length
            if baseline_length > 0.0
            else 0.0
        )

        js = _js_divergence(
            _distribution(baseline),
            _distribution(current),
        )

        metrics = [
            DriftMetric(
                name="unknown_rate",
                baseline=baseline_unknown,
                current=current_unknown,
                delta=unknown_delta,
                threshold=self.unknown_rate_delta_threshold,
                triggered=unknown_delta > self.unknown_rate_delta_threshold,
            ),
            DriftMetric(
                name="mean_confidence",
                baseline=baseline_confidence,
                current=current_confidence,
                delta=confidence_delta,
                threshold=-self.confidence_drop_threshold,
                triggered=confidence_delta < -self.confidence_drop_threshold,
            ),
            DriftMetric(
                name="mean_text_length_relative_change",
                baseline=baseline_length,
                current=current_length,
                delta=length_relative_delta,
                threshold=self.text_length_relative_threshold,
                triggered=abs(length_relative_delta)
                > self.text_length_relative_threshold,
            ),
        ]

        triggered = sum(metric.triggered for metric in metrics)
        relation_triggered = js > self.relation_js_threshold
        total_alerts = triggered + int(relation_triggered)

        if total_alerts >= 2:
            status = "alert"
            message = "Multiple production-distribution drift signals are active."
        elif total_alerts == 1:
            status = "watch"
            message = "One production-distribution drift signal needs review."
        else:
            status = "healthy"
            message = "No configured drift threshold is currently exceeded."

        return DriftSummaryResponse(
            status=status,
            baseline_events=len(baseline),
            current_events=len(current),
            relation_js_divergence=js,
            metrics=metrics,
            message=message,
        )
