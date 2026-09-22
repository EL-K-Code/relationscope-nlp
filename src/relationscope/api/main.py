"""RelationScope HTTP API."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from relationscope.data.schema import PredictRequest, PredictResponse
from relationscope.inference.predictor import RelationPredictor

app = FastAPI(
    title="RelationScope NLP",
    version="0.1.0",
    description="Production Relation Extraction & Knowledge Structuring API",
)


@lru_cache(maxsize=1)
def get_predictor() -> RelationPredictor:
    model_dir = Path(os.getenv("RELATIONSCOPE_MODEL_DIR", "artifacts/transformer/best"))
    device = os.getenv("RELATIONSCOPE_DEVICE", "auto")
    if not model_dir.exists():
        raise RuntimeError(f"model bundle not found: {model_dir}")
    return RelationPredictor(model_dir, device=device)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
