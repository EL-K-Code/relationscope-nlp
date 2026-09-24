"""Active-learning review queue for RelationScope v3."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from relationscope.data.schema import EntitySpan, OpenSetPredictResponse
from relationscope.data.v3_schema import (
    ActiveLearningPolicyResponse,
    ReviewDecisionRequest,
    ReviewPrediction,
    ReviewQueueItem,
    ReviewQueueResponse,
)
from relationscope.ops.store import OperationsStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ActiveLearningService:
    """Score uncertain predictions and persist explicitly consented review examples."""

    def __init__(
        self,
        store: OperationsStore,
        *,
        uncertainty_weight: float = 0.55,
        boundary_weight: float = 0.35,
        unknown_bonus: float = 0.10,
        queue_threshold: float = 0.55,
    ) -> None:
        total = uncertainty_weight + boundary_weight + unknown_bonus
        if abs(total - 1.0) > 1e-6:
            raise ValueError("active-learning policy weights must sum to 1")
        self.store = store
        self.uncertainty_weight = uncertainty_weight
        self.boundary_weight = boundary_weight
        self.unknown_bonus = unknown_bonus
        self.queue_threshold = queue_threshold

    def policy(self) -> ActiveLearningPolicyResponse:
        return ActiveLearningPolicyResponse(
            uncertainty_weight=self.uncertainty_weight,
            boundary_weight=self.boundary_weight,
            unknown_bonus=self.unknown_bonus,
            queue_threshold=self.queue_threshold,
        )

    def score(self, response: OpenSetPredictResponse) -> tuple[float, str]:
        confidence = response.predictions[0].confidence if response.predictions else 0.0
        uncertainty = 1.0 - confidence
        scale = max(response.threshold, 1.0 - response.threshold, 1e-6)
        boundary = 1.0 - min(abs(confidence - response.threshold) / scale, 1.0)
        unknown = 1.0 if response.decision == "UNKNOWN_RELATION" else 0.0

        score = (
            self.uncertainty_weight * uncertainty
            + self.boundary_weight * boundary
            + self.unknown_bonus * unknown
        )
        score = max(0.0, min(1.0, score))

        if response.decision == "UNKNOWN_RELATION":
            reason = "unknown_relation"
        elif abs(confidence - response.threshold) <= 0.05:
            reason = "near_operating_threshold"
        else:
            reason = "low_confidence"
        return score, reason

    def should_queue(self, response: OpenSetPredictResponse) -> bool:
        score, _ = self.score(response)
        return score >= self.queue_threshold

    def enqueue(
        self,
        *,
        text: str,
        head: EntitySpan,
        tail: EntitySpan,
        response: OpenSetPredictResponse,
        source_document_id: str | None,
        consent_to_store_review_data: bool,
    ) -> ReviewQueueItem | None:
        """Queue a review example only when raw-data storage was explicitly allowed."""
        if not consent_to_store_review_data or not self.should_queue(response):
            return None
        if not response.predictions:
            return None

        score, reason = self.score(response)
        now = _utc_now()
        review_id = str(uuid.uuid4())
        top = response.predictions[0]
        item = ReviewQueueItem(
            review_id=review_id,
            created_at=now,
            updated_at=now,
            status="pending",
            active_learning_score=score,
            reason=reason,
            text=text,
            head=head,
            tail=tail,
            prediction=ReviewPrediction(
                relation_id=top.relation_id,
                relation_name=top.relation_name,
                confidence=top.confidence,
                decision=response.decision,
                profile=response.profile,
                threshold=response.threshold,
                model_id=response.model_id,
            ),
            source_document_id=source_document_id,
        )
        self.store.put_json(
            f"review/queue/{review_id}.json",
            item.model_dump(mode="json"),
        )
        return item

    def list_pending(self, *, limit: int = 100) -> ReviewQueueResponse:
        raw_items = self.store.list_json("review/queue", limit=max(limit * 4, limit))
        items = [
            ReviewQueueItem.model_validate(raw)
            for raw in raw_items
            if raw.get("status") == "pending"
        ]
        items.sort(key=lambda item: item.active_learning_score, reverse=True)
        selected = items[:limit]
        return ReviewQueueResponse(items=selected, pending=len(items))

    def get(self, review_id: str) -> ReviewQueueItem | None:
        payload = self.store.get_json(f"review/queue/{review_id}.json")
        return ReviewQueueItem.model_validate(payload) if payload is not None else None

    def review(
        self,
        review_id: str,
        decision: ReviewDecisionRequest,
    ) -> ReviewQueueItem | None:
        item = self.get(review_id)
        if item is None:
            return None

        updated = item.model_copy(
            update={
                "updated_at": _utc_now(),
                "status": "dismissed" if decision.verdict == "dismiss" else "reviewed",
                "reviewer_verdict": decision.verdict,
                "reviewed_relation_id": (
                    decision.corrected_relation_id
                    if decision.verdict == "correct"
                    else item.prediction.relation_id
                    if decision.verdict == "confirm"
                    else None
                ),
                "reviewed_relation_name": (
                    decision.corrected_relation_name
                    if decision.verdict == "correct"
                    else item.prediction.relation_name
                    if decision.verdict == "confirm"
                    else "UNKNOWN_RELATION"
                    if decision.verdict == "unknown"
                    else None
                ),
            }
        )
        self.store.put_json(
            f"review/queue/{review_id}.json",
            updated.model_dump(mode="json"),
        )
        return updated
