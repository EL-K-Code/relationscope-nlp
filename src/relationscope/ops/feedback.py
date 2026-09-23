"""Human feedback capture for RelationScope v2."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from relationscope.data.v2_schema import FeedbackRequest, FeedbackResponse
from relationscope.ops.store import OperationsStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FeedbackService:
    """Persist explicit user feedback with privacy-by-default semantics."""

    def __init__(self, store: OperationsStore) -> None:
        self.store = store

    def submit(self, request: FeedbackRequest) -> FeedbackResponse:
        feedback_id = str(uuid.uuid4())
        created_at = _utc_now()

        payload: dict[str, Any] = {
            "feedback_id": feedback_id,
            "created_at": created_at.isoformat(),
            "verdict": request.verdict,
            "prediction": request.prediction.model_dump(mode="json"),
            "corrected_relation_id": request.corrected_relation_id,
            "corrected_relation_name": request.corrected_relation_name,
            "note": request.note,
            "text_sha256": hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
            "head_sha256": hashlib.sha256(
                request.head.text.encode("utf-8")
            ).hexdigest(),
            "tail_sha256": hashlib.sha256(
                request.tail.text.encode("utf-8")
            ).hexdigest(),
            "consent_to_store_text": request.consent_to_store_text,
        }

        if request.consent_to_store_text:
            payload["text"] = request.text
            payload["head"] = request.head.model_dump(mode="json")
            payload["tail"] = request.tail.model_dump(mode="json")

        self.store.put_json(
            f"feedback/{created_at:%Y/%m/%d}/{feedback_id}.json",
            payload,
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            stored_text=request.consent_to_store_text,
            created_at=created_at,
        )
