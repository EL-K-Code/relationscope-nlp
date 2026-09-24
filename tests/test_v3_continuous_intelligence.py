from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from relationscope.data.schema import (
    BatchPredictExample,
    EntitySpan,
    OpenSetPredictResponse,
    Prediction,
)
from relationscope.data.v2_schema import (
    FeedbackPredictionSnapshot,
    FeedbackRequest,
)
from relationscope.data.v3_schema import (
    DocumentAnalyzeRequest,
    ReviewDecisionRequest,
)
from relationscope.document.analyzer import DocumentAnalyzer, split_sentences
from relationscope.document.entity_extractor import EntityMentionCandidate
from relationscope.ops.active_learning import ActiveLearningService
from relationscope.ops.drift import DriftService
from relationscope.ops.feedback import FeedbackService
from relationscope.ops.store import LocalOperationsStore
from relationscope.training.dataset_versioning import DatasetSnapshotBuilder
from relationscope.training.promotion_gate import PromotionGate


class FakeExtractor:
    model_id = "fake-ner-v1"

    def extract(self, text: str) -> list[EntityMentionCandidate]:
        return [
            EntityMentionCandidate(
                text="Marie Curie",
                start=text.index("Marie Curie"),
                end=text.index("Marie Curie") + len("Marie Curie"),
                label="PER",
            ),
            EntityMentionCandidate(
                text="Warsaw",
                start=text.index("Warsaw"),
                end=text.index("Warsaw") + len("Warsaw"),
                label="LOC",
            ),
        ]


class FakePredictor:
    model_id = "fake-relation-v1"

    def predict_open_set_batch(
        self,
        *,
        examples: list[BatchPredictExample],
        top_k: int,
        profile: str,
    ) -> list[OpenSetPredictResponse]:
        del top_k
        results: list[OpenSetPredictResponse] = []
        for index, example in enumerate(examples):
            known = index == 0
            confidence = 0.99 if known else 0.40
            results.append(
                OpenSetPredictResponse(
                    head=example.head.text,
                    tail=example.tail.text,
                    marked_text=example.text,
                    predictions=[
                        Prediction(
                            relation_id="P19" if known else "P0",
                            relation_name="place of birth" if known else "manufacturer",
                            confidence=confidence,
                        )
                    ],
                    latency_ms=1.0,
                    model_id=self.model_id,
                    decision="KNOWN_RELATION" if known else "UNKNOWN_RELATION",
                    accepted=known,
                    profile=profile,  # type: ignore[arg-type]
                    threshold=0.967,
                    temperature=1.0,
                )
            )
        return results


class MemoryStore:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        del key
        self.events.insert(0, payload)

    def get_json(self, key: str) -> dict[str, Any] | None:
        del key
        return None

    def list_json(self, prefix: str, *, limit: int = 500) -> list[dict[str, Any]]:
        del prefix
        return self.events[:limit]


def test_sentence_split_preserves_offsets() -> None:
    text = "Marie Curie was born in Warsaw. She later worked in Paris."
    sentences = split_sentences(text)
    assert [item.text for item in sentences] == [
        "Marie Curie was born in Warsaw.",
        "She later worked in Paris.",
    ]
    assert text[sentences[1].start : sentences[1].end] == sentences[1].text


def test_document_analyzer_builds_graph_and_queues_uncertain_pair(
    tmp_path: Path,
) -> None:
    store = LocalOperationsStore(tmp_path / "ops")
    active = ActiveLearningService(store)
    analyzer = DocumentAnalyzer(
        predictor=FakePredictor(),  # type: ignore[arg-type]
        entity_extractor=FakeExtractor(),
        active_learning=active,
    )

    text = "Marie Curie was born in Warsaw."
    result = analyzer.analyze(
        DocumentAnalyzeRequest(
            text=text,
            profile="balanced",
            include_unknown=True,
            consent_to_store_review_data=True,
        )
    )

    assert result.model_id == "fake-relation-v1"
    assert result.ner_model_id == "fake-ner-v1"
    assert len(result.entities) == 2
    assert result.candidate_pairs == 2
    assert len(result.relations) == 2
    assert result.review_candidates == 1
    assert result.queued_for_review == 1

    queue = active.list_pending()
    assert queue.pending == 1
    assert queue.items[0].prediction.decision == "UNKNOWN_RELATION"


def test_active_learning_review_becomes_versioned_dataset(
    tmp_path: Path,
) -> None:
    store = LocalOperationsStore(tmp_path / "ops")
    active = ActiveLearningService(store)
    analyzer = DocumentAnalyzer(
        predictor=FakePredictor(),  # type: ignore[arg-type]
        entity_extractor=FakeExtractor(),
        active_learning=active,
    )
    text = "Marie Curie was born in Warsaw."
    analyzer.analyze(
        DocumentAnalyzeRequest(
            text=text,
            profile="balanced",
            include_unknown=True,
            consent_to_store_review_data=True,
        )
    )
    review = active.list_pending().items[0]
    updated = active.review(
        review.review_id,
        ReviewDecisionRequest(
            verdict="correct",
            corrected_relation_id="P19",
            corrected_relation_name="place of birth",
        ),
    )
    assert updated is not None
    assert updated.status == "reviewed"

    feedback = FeedbackService(store)
    feedback.submit(
        FeedbackRequest(
            text=text,
            head=EntitySpan(text="Marie Curie", start=0, end=11),
            tail=EntitySpan(text="Warsaw", start=24, end=30),
            prediction=FeedbackPredictionSnapshot(
                model_id="fake-relation-v1",
                relation_id="P19",
                relation_name="place of birth",
                confidence=0.99,
                decision="KNOWN_RELATION",
                profile="balanced",
                threshold=0.967,
            ),
            verdict="confirm",
            consent_to_store_text=True,
        )
    )

    builder = DatasetSnapshotBuilder(store)
    jsonl_path, manifest_path, manifest = builder.write_snapshot(
        output_dir=tmp_path / "datasets",
        dataset_version="review-v1",
    )

    assert jsonl_path.is_file()
    assert manifest_path.is_file()
    # The active-learning item is the reverse directed pair
    # (Warsaw -> Marie Curie), while feedback confirms
    # (Marie Curie -> Warsaw). Relation extraction is directional, so these
    # are two distinct supervised examples and must not be deduplicated.
    assert manifest.record_count == 2
    assert manifest.relation_counts["place of birth"] == 2
    assert len(manifest.sha256) == 64


def test_drift_service_detects_distribution_shift() -> None:
    now = datetime.now(UTC).isoformat()
    current = [
        {
            "event_type": "prediction",
            "created_at": now,
            "decision": "UNKNOWN_RELATION",
            "confidence": 0.35,
            "text_length": 220,
            "relation_id": "P99",
        }
        for _ in range(20)
    ]
    baseline = [
        {
            "event_type": "prediction",
            "created_at": now,
            "decision": "KNOWN_RELATION",
            "confidence": 0.92,
            "text_length": 80,
            "relation_id": "P19",
        }
        for _ in range(20)
    ]
    service = DriftService(
        MemoryStore(current + baseline),  # type: ignore[arg-type]
        min_events_per_window=20,
    )
    result = service.summary(limit=40)

    assert result.status == "alert"
    assert result.relation_js_divergence > 0.9
    assert any(item.name == "unknown_rate" and item.triggered for item in result.metrics)


def _report(model_version: str) -> dict[str, Any]:
    return {
        "model": {"model_version": model_version},
        "protocol": {
            "name": "fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown"
        },
        "known_test": {"macro_f1": 0.89},
        "final_unseen_test_balanced": {
            "unknown_f1": 0.85,
            "auroc": 0.80,
            "unknown_rejection": 0.84,
        },
        "profiles": {
            "balanced": {
                "accepted_known_accuracy": 0.98,
            }
        },
        "latency": {
            "cloud_run_cpu_warm_ms": {
                "application_median": 90.0,
            }
        },
    }


def test_promotion_gate_accepts_non_regressing_challenger() -> None:
    champion = _report("champion")
    challenger = _report("challenger")
    challenger["known_test"]["macro_f1"] = 0.895
    challenger["final_unseen_test_balanced"]["unknown_f1"] = 0.86
    challenger["latency"]["cloud_run_cpu_warm_ms"]["application_median"] = 100.0

    result = PromotionGate().compare(champion, challenger)
    assert result.passed
    assert all(check.passed for check in result.checks)


def test_promotion_gate_blocks_protocol_change() -> None:
    champion = _report("champion")
    challenger = _report("challenger")
    challenger["protocol"]["name"] = "different-protocol"

    result = PromotionGate().compare(champion, challenger)
    assert not result.passed
    assert not next(
        check for check in result.checks if check.name == "evaluation_protocol"
    ).passed
