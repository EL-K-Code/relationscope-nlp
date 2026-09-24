"""RelationScope HTTP API."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from pypdf import PdfReader

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
    OpenSetProfile,
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
    RelationCatalogResponse,
)
from relationscope.data.v3_schema import (
    ActiveLearningPolicyResponse,
    CanaryStatusResponse,
    ChallengerRegistrationRequest,
    ChallengerRegistrationResponse,
    DocumentAnalyzeRequest,
    DocumentAnalyzeResponse,
    DriftSummaryResponse,
    LearningSummaryResponse,
    ReviewDecisionRequest,
    ReviewQueueItem,
    ReviewQueueResponse,
)
from relationscope.document.analyzer import DocumentAnalyzer
from relationscope.document.entity_extractor import build_entity_extractor
from relationscope.inference.model_store import resolve_model_bundle
from relationscope.inference.predictor import RelationPredictor
from relationscope.ops.active_learning import ActiveLearningService
from relationscope.ops.batch import BatchJobService, dispatcher_from_env
from relationscope.ops.challenger import ChallengerService, canary_status
from relationscope.ops.drift import DriftService
from relationscope.ops.feedback import FeedbackService
from relationscope.ops.registry import registry_from_env
from relationscope.ops.store import get_operations_store
from relationscope.ops.telemetry import TelemetryService

load_dotenv(override=False)

logger = logging.getLogger("relationscope.api")

app = FastAPI(
    title="RelationScope NLP",
    version="3.0.0",
    description=(
        "Production relation intelligence API with open-set inference, "
        "document analysis and continuous-learning operations"
    ),
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


@lru_cache(maxsize=1)
def get_active_learning_service() -> ActiveLearningService:
    return ActiveLearningService(get_operations_store())


@lru_cache(maxsize=1)
def get_drift_service() -> DriftService:
    return DriftService(get_operations_store())


@lru_cache(maxsize=1)
def get_challenger_service() -> ChallengerService:
    return ChallengerService(get_operations_store())


@lru_cache(maxsize=1)
def get_document_analyzer() -> DocumentAnalyzer:
    return DocumentAnalyzer(
        predictor=get_predictor(),
        entity_extractor=build_entity_extractor(),
        active_learning=get_active_learning_service(),
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
            relation_id=response.predictions[0].relation_id,
            relation_name=response.predictions[0].relation_name,
            deployment_variant=os.getenv(
                "RELATIONSCOPE_DEPLOYMENT_VARIANT",
                "champion",
            ),
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


@app.get("/v2/relations", response_model=RelationCatalogResponse)
def relation_catalog() -> RelationCatalogResponse:
    try:
        return get_predictor().relation_catalog()
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
def create_batch_job(
    request: BatchJobCreateRequest,
    http_request: Request,
) -> BatchJobSummary:
    try:
        api_base_url = str(http_request.base_url).rstrip("/")
        return get_batch_service().create(request, api_base_url=api_base_url)
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


# ---------------------------------------------------------------------------
# V3 API — automatic document intelligence and continuous learning.
# ---------------------------------------------------------------------------


@app.post("/v3/documents:analyze", response_model=DocumentAnalyzeResponse)
def analyze_document(request: DocumentAnalyzeRequest) -> DocumentAnalyzeResponse:
    try:
        return get_document_analyzer().analyze(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v3/documents:upload", response_model=DocumentAnalyzeResponse)
async def upload_document(
    file: UploadFile = File(...),
    profile: OpenSetProfile = Form("balanced"),
    include_unknown: bool = Form(False),
    max_candidate_pairs: int = Form(256),
    consent_to_store_review_data: bool = Form(False),
) -> DocumentAnalyzeResponse:
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="document exceeds the 10 MiB limit")

    filename = file.filename or "document"
    content_type = (file.content_type or "").lower()

    try:
        if filename.lower().endswith(".pdf") or content_type == "application/pdf":
            reader = PdfReader(BytesIO(raw))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = raw.decode("utf-8")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="unable to extract UTF-8 text from the uploaded document",
        ) from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="document contains no extractable text")

    request = DocumentAnalyzeRequest(
        text=text,
        title=filename,
        profile=profile,
        include_unknown=include_unknown,
        max_candidate_pairs=max_candidate_pairs,
        consent_to_store_review_data=consent_to_store_review_data,
    )
    return analyze_document(request)


@app.get(
    "/v3/active-learning/policy",
    response_model=ActiveLearningPolicyResponse,
)
def active_learning_policy() -> ActiveLearningPolicyResponse:
    return get_active_learning_service().policy()


@app.get("/v3/drift", response_model=DriftSummaryResponse)
def drift_summary(
    limit: int = Query(default=400, ge=40, le=5_000),
) -> DriftSummaryResponse:
    return get_drift_service().summary(limit=limit)


@app.get("/v3/canary", response_model=CanaryStatusResponse)
def get_canary_status() -> CanaryStatusResponse:
    return canary_status(
        champion_model_id=get_predictor().model_id,
        challenger_service=get_challenger_service(),
    )


@app.get("/v3/learning/summary", response_model=LearningSummaryResponse)
def learning_summary() -> LearningSummaryResponse:
    queue = get_operations_store().list_json("review/queue", limit=10_000)
    pending = sum(item.get("status") == "pending" for item in queue)
    reviewed = sum(item.get("status") == "reviewed" for item in queue)
    dismissed = sum(item.get("status") == "dismissed" for item in queue)
    return LearningSummaryResponse(
        pending_review=pending,
        reviewed_examples=reviewed,
        dismissed_examples=dismissed,
        drift=get_drift_service().summary(limit=400),
        canary=get_canary_status(),
    )


@app.get(
    "/v3/internal/review-queue",
    response_model=ReviewQueueResponse,
)
def internal_review_queue(
    limit: int = Query(default=100, ge=1, le=500),
) -> ReviewQueueResponse:
    return get_active_learning_service().list_pending(limit=limit)


@app.post(
    "/v3/internal/review-queue/{review_id}:review",
    response_model=ReviewQueueItem,
)
def internal_review_decision(
    review_id: str,
    decision: ReviewDecisionRequest,
) -> ReviewQueueItem:
    item = get_active_learning_service().review(review_id, decision)
    if item is None:
        raise HTTPException(status_code=404, detail="review item not found")
    return item


@app.post(
    "/v3/internal/challengers",
    response_model=ChallengerRegistrationResponse,
    status_code=201,
)
def internal_register_challenger(
    request: ChallengerRegistrationRequest,
) -> ChallengerRegistrationResponse:
    return get_challenger_service().register(request)
