"""Typed data models used by RelationScope."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntitySpan(BaseModel):
    """Character span in a source text. End is exclusive."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "EntitySpan":
        if self.end <= self.start:
            raise ValueError("entity end must be greater than start")
        return self


class RelationRecord(BaseModel):
    """Canonical relation-extraction example."""

    relation_id: str
    relation_name: str
    relation_description: str
    tokens: list[str]
    marked_text: str
    head_text: str
    tail_text: str
    source_split: str


class Prediction(BaseModel):
    relation_id: str
    relation_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionItem(BaseModel):
    head: str
    tail: str
    marked_text: str
    predictions: list[Prediction]


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    head: EntitySpan
    tail: EntitySpan
    top_k: int = Field(default=5, ge=1, le=20)


class PredictResponse(PredictionItem):
    latency_ms: float
    model_id: str


class BatchPredictExample(BaseModel):
    text: str = Field(min_length=1)
    head: EntitySpan
    tail: EntitySpan


class BatchPredictRequest(BaseModel):
    examples: list[BatchPredictExample] = Field(min_length=1, max_length=128)
    top_k: int = Field(default=5, ge=1, le=20)


class BatchPredictResponse(BaseModel):
    items: list[PredictionItem]
    latency_ms: float
    model_id: str
    batch_size: int


class ModelInfoResponse(BaseModel):
    model_id: str
    device: str
    num_relations: int
    max_length: int
