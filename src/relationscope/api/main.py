"""RelationScope HTTP API."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from relationscope.data.schema import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    OpenSetPredictResponse,
    PredictRequest,
    PredictResponse,
)
from relationscope.inference.model_store import resolve_model_bundle
from relationscope.inference.predictor import RelationPredictor

app = FastAPI(
    title="RelationScope NLP",
    version="0.1.0",
    description="Production Relation Extraction & Knowledge Structuring API",
)


@lru_cache(maxsize=1)
def get_predictor() -> RelationPredictor:
    model_dir = resolve_model_bundle()
    device = os.getenv("RELATIONSCOPE_DEVICE", "auto")
    if not model_dir.exists():
        raise RuntimeError(
            "model bundle not found locally and no usable GCS bundle was configured"
        )
    return RelationPredictor(model_dir, device=device)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe that deliberately does not force model loading."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness probe: succeeds only after a model bundle can be loaded."""
    try:
        predictor = get_predictor()
        return {"status": "ready", "model_id": predictor.model_id}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return get_predictor().info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/relations:predict", response_model=PredictResponse)
def predict_relation(request: PredictRequest) -> PredictResponse:
    try:
        return get_predictor().predict(
            text=request.text,
            head=request.head,
            tail=request.tail,
            top_k=request.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/relations:predict-open-set", response_model=OpenSetPredictResponse)
def predict_relation_open_set(request: PredictRequest) -> OpenSetPredictResponse:
    try:
        return get_predictor().predict_open_set(
            text=request.text,
            head=request.head,
            tail=request.tail,
            top_k=request.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/relations:explain", response_model=ExplainResponse)
def explain_relation(request: ExplainRequest) -> ExplainResponse:
    try:
        return get_predictor().explain(
            text=request.text,
            head=request.head,
            tail=request.tail,
            top_evidence=request.top_evidence,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/relations:predict-batch", response_model=BatchPredictResponse)
def predict_relations_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    try:
        return get_predictor().predict_batch(
            examples=request.examples,
            top_k=request.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
