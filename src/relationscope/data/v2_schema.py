"""Typed API contracts introduced in RelationScope v2."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from relationscope.data.schema import (
    BatchPredictExample,
    EntitySpan,
    OpenSetProfile,
    Prediction,
)


class InfluenceExplainRequest(BaseModel):
    """Request a relation-specific word influence map."""

    text: str = Field(min_length=1)
    head: EntitySpan
    tail: EntitySpan
    profile: OpenSetProfile = "balanced"
    target_relation_id: str | None = None


class WordInfluence(BaseModel):
    """Effect of masking one complete input word on target confidence."""

    word: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence_drop: float
    normalized_influence: float = Field(ge=-1.0, le=1.0)
    direction: Literal["supports", "opposes", "neutral"]


class InfluenceExplainResponse(BaseModel):
    model_id: str
    relation: Prediction
    decision: Literal["KNOWN_RELATION", "UNKNOWN_RELATION"]
    profile: OpenSetProfile
    threshold: float = Field(ge=0.0, le=1.0)
    baseline_confidence: float = Field(ge=0.0, le=1.0)
    marked_text: str
    influences: list[WordInfluence]
    method: Literal["word_occlusion"] = "word_occlusion"
    latency_ms: float


class FeedbackPredictionSnapshot(BaseModel):
    model_id: str
    relation_id: str
    relation_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision: Literal["KNOWN_RELATION", "UNKNOWN_RELATION"]
    profile: OpenSetProfile
    threshold: float = Field(ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    """Explicit user feedback. Raw text is persisted only with opt-in consent."""

    text: str = Field(min_length=1)
    head: EntitySpan
    tail: EntitySpan
    prediction: FeedbackPredictionSnapshot
    verdict: Literal["confirm", "correct", "unknown", "skip"]
    corrected_relation_id: str | None = None
    corrected_relation_name: str | None = None
    note: str | None = Field(default=None, max_length=1000)
    consent_to_store_text: bool = False

    @model_validator(mode="after")
    def validate_correction(self) -> FeedbackRequest:
        if self.verdict == "correct" and not self.corrected_relation_name:
            raise ValueError("corrected_relation_name is required for corrected feedback")
        return self


class FeedbackResponse(BaseModel):
    feedback_id: str
    stored_text: bool
    created_at: datetime


class BatchJobCreateRequest(BaseModel):
    examples: list[BatchPredictExample] = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=1, ge=1, le=20)
    profile: OpenSetProfile = "balanced"


class BatchJobSummary(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    model_id: str | None = None
    error: str | None = None


class BatchJobItem(BaseModel):
    head: str
    tail: str
    relation: Prediction
    decision: Literal["KNOWN_RELATION", "UNKNOWN_RELATION"]
    accepted: bool
    profile: OpenSetProfile
    threshold: float


class BatchJobResponse(BatchJobSummary):
    results: list[BatchJobItem] | None = None


class TelemetryProfileCount(BaseModel):
    profile: str
    count: int = Field(ge=0)


class TelemetryModelCount(BaseModel):
    model_id: str
    count: int = Field(ge=0)


class MetricsSummaryResponse(BaseModel):
    window_events: int = Field(ge=0)
    total_predictions: int = Field(ge=0)
    known_rate: float = Field(ge=0.0, le=1.0)
    unknown_rate: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    profiles: list[TelemetryProfileCount]
    models: list[TelemetryModelCount]


class ModelRegistryEntry(BaseModel):
    model_version: str
    gcs_prefix: str
    promoted: bool = False
    created_at: datetime | None = None


class ModelRegistryResponse(BaseModel):
    current_model_id: str
    entries: list[ModelRegistryEntry]
