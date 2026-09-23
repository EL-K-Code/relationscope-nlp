"""RelationScope HTTP API."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from relationscope.api.observability import RequestLoggingMiddleware
from relationscope.data.schema import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    OpenSetPredictRequest,
    OpenSetPredictResponse,
    PredictRequest,
    PredictResponse,
)
from relationscope.data.v2_schema import (
    BatchJobCreateRequest,
    BatchJobResponse,
    BatchJobSummary,
    FeedbackRequest,
    FeedbackResponse,
    InfluenceExplainRequest,
    InfluenceExplainResponse,
    MetricsSummaryResponse,
    ModelRegistryResponse,
)
from relationscope.inference.model_store import resolve_model_bundle
from relationscope.inference.predictor import RelationPredictor
from relationscope.ops.batch import BatchJobService, dispatcher_from_env
from relationscope.ops.feedback import FeedbackService
from relationscope.ops.registry import registry_from_env
from relationscope.ops.store import get_operations_store
from relationscope.ops.telemetry import TelemetryService

load_dotenv(override=False)

logger = logging.getLogger("relationscope.api")

app = FastAPI(
    title="RelationScope NLP",
    version="2.0.0",
    description="Production relation extraction, open-set inference and ML operations API",
)
app.add_middleware(RequestLoggingMiddleware)


@lru_cache(maxsize=1)
def get_predictor() -> RelationPredictor:
    model_dir = resolve_model_bundle()
    device = os.getenv("RELATIONSCOPE_DEVICE", "auto")
    if not model_dir.exists():
        raise RuntimeError(
            "model bundle not found locally and no usable GCS bundle was configured"
        )
    return RelationPredictor(model_dir, device=device)


@lru_cache(maxsize=1)
def get_telemetry_service() -> TelemetryService:
    return TelemetryService(get_operations_store())


@lru_cache(maxsize=1)
def get_feedback_service() -> FeedbackService:
    return FeedbackService(get_operations_store())


@lru_cache(maxsize=1)
def get_batch_service() -> BatchJobService:
    return BatchJobService(
        get_operations_store(),
        get_predictor,
        dispatcher=dispatcher_from_env(),
    )


def _record_prediction_safely(
    request: OpenSetPredictRequest,
    response: OpenSetPredictResponse,
) -> None:
    try:
        get_telemetry_service().record_prediction(
            text=request.text,
            model_id=response.model_id,
            profile=response.profile,
            decision=response.decision,
            confidence=response.predictions[0].confidence,
            threshold=response.threshold,
            latency_ms=response.latency_ms,
        )
    except Exception as exc:
        # Telemetry must never make inference unavailable.
        logger.warning("prediction telemetry write failed: %s", exc)


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


# ---------------------------------------------------------------------------
# V1 compatibility API. Frozen for clients built against RelationScope v1.0.
# ---------------------------------------------------------------------------


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
def predict_relation_open_set(request: OpenSetPredictRequest) -> OpenSetPredictResponse:
    try:
        return get_predictor().predict_open_set(
            text=request.text,
            head=request.head,
            tail=request.tail,
            top_k=request.top_k,
            profile=request.profile,
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


# ---------------------------------------------------------------------------
# V2 API.
# ---------------------------------------------------------------------------


@app.get("/v2/model", response_model=ModelInfoResponse)
def model_info_v2() -> ModelInfoResponse:
    return model_info()


@app.post("/v2/relations:predict-open-set", response_model=OpenSetPredictResponse)
def predict_relation_open_set_v2(
    request: OpenSetPredictRequest,
) -> OpenSetPredictResponse:
    try:
        response = get_predictor().predict_open_set(
            text=request.text,
            head=request.head,
            tail=request.tail,
            top_k=request.top_k,
            profile=request.profile,
        )
        _record_prediction_safely(request, response)
        return response
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v2/relations:explain", response_model=InfluenceExplainResponse)
def explain_relation_v2(
    request: InfluenceExplainRequest,
) -> InfluenceExplainResponse:
    try:
        return get_predictor().explain_words(
            text=request.text,
            head=request.head,
            tail=request.tail,
            profile=request.profile,
            target_relation_id=request.target_relation_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v2/feedback", response_model=FeedbackResponse, status_code=201)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        return get_feedback_service().submit(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v2/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    limit: int = Query(default=500, ge=1, le=5000),
) -> MetricsSummaryResponse:
    try:
        return get_telemetry_service().summary(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v2/models", response_model=ModelRegistryResponse)
def model_registry() -> ModelRegistryResponse:
    try:
        predictor = get_predictor()
        return registry_from_env(predictor.model_id).list()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v2/batch-jobs", response_model=BatchJobSummary, status_code=202)
def create_batch_job(request: BatchJobCreateRequest) -> BatchJobSummary:
    try:
        return get_batch_service().create(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v2/batch-jobs/{job_id}", response_model=BatchJobResponse)
def get_batch_job(job_id: str) -> BatchJobResponse:
    try:
        result = get_batch_service().get(job_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="batch job not found")
    return result


@app.post(
    "/v2/internal/batch-jobs/{job_id}:run",
    response_model=BatchJobSummary,
)
def run_batch_job(job_id: str) -> BatchJobSummary:
    """Cloud Tasks callback. The Cloud Run service itself remains IAM protected."""
    try:
        return get_batch_service().run(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="batch job not found") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
