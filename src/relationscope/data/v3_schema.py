"""Typed API contracts introduced in RelationScope v3."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from relationscope.data.schema import EntitySpan, OpenSetProfile, Prediction
from relationscope.data.v2_schema import FeedbackPredictionSnapshot


class DocumentAnalyzeRequest(BaseModel):
    """Analyze free text without requiring caller-supplied entity spans."""

    text: str = Field(min_length=1, max_length=200_000)
    title: str | None = Field(default=None, max_length=500)
    profile: OpenSetProfile = "balanced"
    include_unknown: bool = False
    max_candidate_pairs: int = Field(default=256, ge=1, le=2_000)
    consent_to_store_review_data: bool = False


class EntityMention(BaseModel):
    sentence_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class DocumentEntity(BaseModel):
    entity_id: str
    text: str
    normalized_text: str
    label: str
    mentions: list[EntityMention]


class DocumentRelation(BaseModel):
    edge_id: str
    head_entity_id: str
    tail_entity_id: str
    head_text: str
    tail_text: str
    relation: Prediction
    decision: Literal["KNOWN_RELATION", "UNKNOWN_RELATION"]
    accepted: bool
    profile: OpenSetProfile
    threshold: float = Field(ge=0.0, le=1.0)
    sentence_index: int = Field(ge=0)
    evidence_text: str


class DocumentAnalyzeResponse(BaseModel):
    document_id: str
    title: str | None = None
    model_id: str
    ner_model_id: str
    entities: list[DocumentEntity]
    relations: list[DocumentRelation]
    candidate_pairs: int = Field(ge=0)
    processed_pairs: int = Field(ge=0)
    review_candidates: int = Field(ge=0)
    queued_for_review: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class ReviewPrediction(BaseModel):
    relation_id: str
    relation_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision: Literal["KNOWN_RELATION", "UNKNOWN_RELATION"]
    profile: OpenSetProfile
    threshold: float = Field(ge=0.0, le=1.0)
    model_id: str


class ReviewQueueItem(BaseModel):
    review_id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["pending", "reviewed", "dismissed"]
    active_learning_score: float = Field(ge=0.0, le=1.0)
    reason: str
    text: str
    head: EntitySpan
    tail: EntitySpan
    prediction: ReviewPrediction
    source_document_id: str | None = None
    reviewed_relation_id: str | None = None
    reviewed_relation_name: str | None = None
    reviewer_verdict: Literal["confirm", "correct", "unknown", "dismiss"] | None = None


class ReviewQueueResponse(BaseModel):
    items: list[ReviewQueueItem]
    pending: int = Field(ge=0)


class ReviewDecisionRequest(BaseModel):
    verdict: Literal["confirm", "correct", "unknown", "dismiss"]
    corrected_relation_id: str | None = None
    corrected_relation_name: str | None = None

    @model_validator(mode="after")
    def validate_correction(self) -> ReviewDecisionRequest:
        if self.verdict == "correct" and not self.corrected_relation_name:
            raise ValueError("corrected_relation_name is required for corrected review")
        return self


class ActiveLearningPolicyResponse(BaseModel):
    uncertainty_weight: float
    boundary_weight: float
    unknown_bonus: float
    queue_threshold: float


class DriftMetric(BaseModel):
    name: str
    baseline: float
    current: float
    delta: float
    threshold: float
    triggered: bool


class DriftSummaryResponse(BaseModel):
    status: Literal["insufficient_data", "healthy", "watch", "alert"]
    baseline_events: int = Field(ge=0)
    current_events: int = Field(ge=0)
    relation_js_divergence: float = Field(ge=0.0)
    metrics: list[DriftMetric]
    message: str


class CanaryStatusResponse(BaseModel):
    enabled: bool
    traffic_percent: float = Field(ge=0.0, le=100.0)
    champion_model_id: str
    challenger_url_configured: bool
    challenger_model_version: str | None = None


class DatasetSnapshotManifest(BaseModel):
    schema_version: Literal["3.0"] = "3.0"
    dataset_version: str
    created_at: datetime
    record_count: int = Field(ge=0)
    sha256: str
    source_counts: dict[str, int]
    relation_counts: dict[str, int]


class PromotionGateCheck(BaseModel):
    name: str
    champion: float | str
    challenger: float | str
    tolerance: float | None = None
    passed: bool
    detail: str


class PromotionGateResult(BaseModel):
    champion_model: str
    challenger_model: str
    passed: bool
    checks: list[PromotionGateCheck]
    warnings: list[str] = Field(default_factory=list)


class ChallengerRegistrationRequest(BaseModel):
    """Metadata-only challenger declaration for dashboards and audit."""

    model_version: str = Field(min_length=1)
    evaluation_report_uri: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    notes: str | None = Field(default=None, max_length=2_000)


class ChallengerRegistrationResponse(BaseModel):
    challenger_id: str
    created_at: datetime
    model_version: str
    dataset_version: str
    status: Literal["registered"] = "registered"


class TrainingCandidate(BaseModel):
    """Reviewed example exported to a versioned training snapshot."""

    text: str
    head: EntitySpan
    tail: EntitySpan
    relation_id: str | None = None
    relation_name: str
    source: Literal["feedback", "active_learning"]
    source_id: str
