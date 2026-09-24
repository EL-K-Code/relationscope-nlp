"""Champion/challenger metadata and canary status for RelationScope v3."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

from relationscope.data.v3_schema import (
    CanaryStatusResponse,
    ChallengerRegistrationRequest,
    ChallengerRegistrationResponse,
)
from relationscope.ops.store import OperationsStore


class ChallengerService:
    def __init__(self, store: OperationsStore) -> None:
        self.store = store

    def register(
        self,
        request: ChallengerRegistrationRequest,
    ) -> ChallengerRegistrationResponse:
        created_at = datetime.now(UTC)
        challenger_id = str(uuid.uuid4())
        payload = {
            "challenger_id": challenger_id,
            "created_at": created_at.isoformat(),
            "model_version": request.model_version,
            "dataset_version": request.dataset_version,
            "evaluation_report_uri": request.evaluation_report_uri,
            "notes": request.notes,
            "status": "registered",
        }
        self.store.put_json(
            f"challengers/{created_at:%Y/%m/%d}/{challenger_id}.json",
            payload,
        )
        return ChallengerRegistrationResponse(
            challenger_id=challenger_id,
            created_at=created_at,
            model_version=request.model_version,
            dataset_version=request.dataset_version,
        )

    def latest_model_version(self) -> str | None:
        items = self.store.list_json("challengers", limit=1)
        if not items:
            return None
        value = items[0].get("model_version")
        return str(value) if value else None


def canary_status(
    *,
    champion_model_id: str,
    challenger_service: ChallengerService,
) -> CanaryStatusResponse:
    raw_percent = os.getenv("RELATIONSCOPE_CANARY_PERCENT", "0")
    try:
        percent = float(raw_percent)
    except ValueError:
        percent = 0.0
    percent = max(0.0, min(100.0, percent))

    challenger_url = bool(os.getenv("RELATIONSCOPE_CHALLENGER_API_URL", "").strip())
    enabled = challenger_url and percent > 0.0

    return CanaryStatusResponse(
        enabled=enabled,
        traffic_percent=percent,
        champion_model_id=champion_model_id,
        challenger_url_configured=challenger_url,
        challenger_model_version=challenger_service.latest_model_version(),
    )
