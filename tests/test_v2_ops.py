from __future__ import annotations

from pathlib import Path

from relationscope.data.schema import (
    BatchPredictExample,
    EntitySpan,
    OpenSetPredictResponse,
    Prediction,
)
from relationscope.data.v2_schema import (
    BatchJobCreateRequest,
    FeedbackPredictionSnapshot,
    FeedbackRequest,
)
from relationscope.ops.batch import BatchJobService
from relationscope.ops.feedback import FeedbackService
from relationscope.ops.store import LocalOperationsStore
from relationscope.ops.telemetry import TelemetryService


class FakePredictor:
    model_id = "test-model-v2"

    def predict_open_set_batch(
        self,
        *,
        examples: list[BatchPredictExample],
        top_k: int,
        profile: str,
    ) -> list[OpenSetPredictResponse]:
        del top_k
        return [
            OpenSetPredictResponse(
                head=example.head.text,
                tail=example.tail.text,
                marked_text=example.text,
                predictions=[
                    Prediction(
                        relation_id="P19",
                        relation_name="place of birth",
                        confidence=0.99,
                    )
                ],
                latency_ms=1.0,
                model_id=self.model_id,
                decision="KNOWN_RELATION",
                accepted=True,
                profile=profile,  # type: ignore[arg-type]
                threshold=0.967,
                temperature=1.0,
            )
            for example in examples
        ]


def _store(tmp_path: Path) -> LocalOperationsStore:
    return LocalOperationsStore(tmp_path / "ops")


def _example() -> BatchPredictExample:
    text = "Marie Curie was born in Warsaw."
    return BatchPredictExample(
        text=text,
        head=EntitySpan(text="Marie Curie", start=0, end=11),
        tail=EntitySpan(text="Warsaw", start=24, end=30),
    )


def test_telemetry_summary_is_privacy_preserving(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = TelemetryService(store)

    service.record_prediction(
        text="private sentence one",
        model_id="m1",
        profile="balanced",
        decision="KNOWN_RELATION",
        confidence=0.98,
        threshold=0.967,
        latency_ms=80.0,
    )
    service.record_prediction(
        text="private sentence two",
        model_id="m1",
        profile="balanced",
        decision="UNKNOWN_RELATION",
        confidence=0.40,
        threshold=0.967,
        latency_ms=120.0,
    )

    summary = service.summary()
    assert summary.total_predictions == 2
    assert summary.known_rate == 0.5
    assert summary.unknown_rate == 0.5
    assert summary.p50_latency_ms == 100.0

    persisted = store.list_json("telemetry")
    assert all("text" not in event for event in persisted)
    assert all("text_sha256" in event for event in persisted)


def test_feedback_requires_explicit_consent_for_raw_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = FeedbackService(store)
    example = _example()
    prediction = FeedbackPredictionSnapshot(
        model_id="m1",
        relation_id="P19",
        relation_name="place of birth",
        confidence=0.98,
        decision="KNOWN_RELATION",
        profile="balanced",
        threshold=0.967,
    )

    response = service.submit(
        FeedbackRequest(
            text=example.text,
            head=example.head,
            tail=example.tail,
            prediction=prediction,
            verdict="confirm",
            consent_to_store_text=False,
        )
    )

    assert response.stored_text is False
    saved = store.list_json("feedback")
    assert len(saved) == 1
    assert "text" not in saved[0]
    assert "head" not in saved[0]
    assert "tail" not in saved[0]


def test_local_batch_job_executes_and_persists_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fake = FakePredictor()
    service = BatchJobService(
        store,
        lambda: fake,  # type: ignore[arg-type]
        dispatcher=None,
        chunk_size=1,
    )

    summary = service.create(
        BatchJobCreateRequest(
            examples=[_example(), _example()],
            top_k=1,
            profile="balanced",
        )
    )

    assert summary.status == "succeeded"
    assert summary.processed == 2
    assert summary.model_id == fake.model_id

    job = service.get(summary.job_id)
    assert job is not None
    assert job.status == "succeeded"
    assert job.results is not None
    assert len(job.results) == 2
    assert job.results[0].relation.relation_name == "place of birth"
