"""Privacy-preserving telemetry aggregation for RelationScope v2."""

from __future__ import annotations

import hashlib
import math
import statistics
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from relationscope.data.v2_schema import (
    MetricsSummaryResponse,
    TelemetryModelCount,
    TelemetryProfileCount,
)
from relationscope.ops.store import OperationsStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class TelemetryService:
    def __init__(self, store: OperationsStore) -> None:
        self.store = store

    def record_prediction(
        self,
        *,
        text: str,
        model_id: str,
        profile: str,
        decision: str,
        confidence: float,
        threshold: float,
        latency_ms: float,
        relation_id: str | None = None,
        relation_name: str | None = None,
        deployment_variant: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        now = _utc_now()
        payload: dict[str, Any] = {
            "event_id": event_id,
            "event_type": "prediction",
            "created_at": now.isoformat(),
            "model_id": model_id,
            "profile": profile,
            "decision": decision,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "latency_ms": float(latency_ms),
            "relation_id": relation_id,
            "relation_name": relation_name,
            "deployment_variant": deployment_variant,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_length": len(text),
        }
        self.store.put_json(
            f"telemetry/{now:%Y/%m/%d}/{now:%H}/{event_id}.json",
            payload,
        )
        return event_id

    def summary(self, *, limit: int = 500) -> MetricsSummaryResponse:
        events = self.store.list_json("telemetry", limit=limit)
        predictions = [
            event for event in events if event.get("event_type") == "prediction"
        ]

        if not predictions:
            return MetricsSummaryResponse(
                window_events=0,
                total_predictions=0,
                known_rate=0.0,
                unknown_rate=0.0,
                mean_confidence=0.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                profiles=[],
                models=[],
            )

        latencies = [float(event.get("latency_ms", 0.0)) for event in predictions]
        confidences = [float(event.get("confidence", 0.0)) for event in predictions]
        known = sum(event.get("decision") == "KNOWN_RELATION" for event in predictions)
        total = len(predictions)

        profile_counts = Counter(str(event.get("profile", "unknown")) for event in predictions)
        model_counts = Counter(str(event.get("model_id", "unknown")) for event in predictions)

        return MetricsSummaryResponse(
            window_events=len(events),
            total_predictions=total,
            known_rate=known / total,
            unknown_rate=(total - known) / total,
            mean_confidence=statistics.fmean(confidences),
            mean_latency_ms=statistics.fmean(latencies),
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            profiles=[
                TelemetryProfileCount(profile=name, count=count)
                for name, count in profile_counts.most_common()
            ],
            models=[
                TelemetryModelCount(model_id=name, count=count)
                for name, count in model_counts.most_common()
            ],
        )
