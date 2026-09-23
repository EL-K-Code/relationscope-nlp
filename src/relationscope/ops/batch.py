"""Durable asynchronous batch-job orchestration for RelationScope v2."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from relationscope.data.schema import BatchPredictExample
from relationscope.data.v2_schema import (
    BatchJobCreateRequest,
    BatchJobItem,
    BatchJobResponse,
    BatchJobSummary,
)
from relationscope.inference.predictor import RelationPredictor
from relationscope.ops.store import OperationsStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _job_key(job_id: str) -> str:
    return f"batch/jobs/{job_id}.json"


def _input_key(job_id: str) -> str:
    return f"batch/inputs/{job_id}.json"


def _result_key(job_id: str) -> str:
    return f"batch/results/{job_id}.json"


class BatchDispatcher(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class CloudTasksDispatcher:
    """Dispatch a private authenticated callback through Google Cloud Tasks."""

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        queue: str,
        target_base_url: str,
        service_account_email: str,
    ) -> None:
        from google.cloud import tasks_v2

        self.tasks_v2 = tasks_v2
        self.client = tasks_v2.CloudTasksClient()
        self.parent = self.client.queue_path(project_id, location, queue)
        self.target_base_url = target_base_url.rstrip("/")
        self.service_account_email = service_account_email

    def enqueue(self, job_id: str) -> None:
        target_url = f"{self.target_base_url}/{job_id}:run"
        task = {
            "http_request": {
                "http_method": self.tasks_v2.HttpMethod.POST,
                "url": target_url,
                "headers": {"Content-Type": "application/json"},
                "oidc_token": {
                    "service_account_email": self.service_account_email,
                    "audience": os.environ["RELATIONSCOPE_API_AUDIENCE"],
                },
            },
            "dispatch_deadline": {"seconds": 900},
        }
        self.client.create_task(parent=self.parent, task=task)


def dispatcher_from_env() -> BatchDispatcher | None:
    required = {
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("RELATIONSCOPE_TASK_LOCATION"),
        "queue": os.getenv("RELATIONSCOPE_TASK_QUEUE"),
        "target_base_url": os.getenv("RELATIONSCOPE_TASK_TARGET_URL"),
        "service_account_email": os.getenv("RELATIONSCOPE_TASK_SERVICE_ACCOUNT"),
    }
    if not all(required.values()):
        return None
    return CloudTasksDispatcher(**required)  # type: ignore[arg-type]


class BatchJobService:
    def __init__(
        self,
        store: OperationsStore,
        predictor_factory: Callable[[], RelationPredictor],
        *,
        dispatcher: BatchDispatcher | None = None,
        chunk_size: int = 64,
    ) -> None:
        self.store = store
        self.predictor_factory = predictor_factory
        self.dispatcher = dispatcher
        self.chunk_size = chunk_size

    def create(self, request: BatchJobCreateRequest) -> BatchJobSummary:
        job_id = str(uuid.uuid4())
        now = _utc_now()
        status = BatchJobSummary(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            total=len(request.examples),
            processed=0,
        )
        self.store.put_json(_input_key(job_id), request.model_dump(mode="json"))
        self.store.put_json(_job_key(job_id), status.model_dump(mode="json"))

        if self.dispatcher is not None:
            try:
                self.dispatcher.enqueue(job_id)
            except Exception:
                failed = status.model_copy(
                    update={
                        "status": "failed",
                        "updated_at": _utc_now(),
                        "error": "failed to enqueue Cloud Task",
                    }
                )
                self.store.put_json(_job_key(job_id), failed.model_dump(mode="json"))
                raise
        else:
            # Local-development fallback: execute synchronously while preserving
            # exactly the same durable job records.
            self.run(job_id)

        payload = self.store.get_json(_job_key(job_id))
        if payload is None:
            raise RuntimeError(f"batch job disappeared after creation: {job_id}")
        return BatchJobSummary.model_validate(payload)

    def get(self, job_id: str) -> BatchJobResponse | None:
        status_payload = self.store.get_json(_job_key(job_id))
        if status_payload is None:
            return None

        status = BatchJobSummary.model_validate(status_payload)
        results: list[BatchJobItem] | None = None
        if status.status == "succeeded":
            result_payload = self.store.get_json(_result_key(job_id))
            if result_payload is not None:
                raw_items = result_payload.get("items", [])
                results = [BatchJobItem.model_validate(item) for item in raw_items]

        return BatchJobResponse(**status.model_dump(), results=results)

    def run(self, job_id: str) -> BatchJobSummary:
        status_payload = self.store.get_json(_job_key(job_id))
        input_payload = self.store.get_json(_input_key(job_id))
        if status_payload is None or input_payload is None:
            raise KeyError(job_id)

        status = BatchJobSummary.model_validate(status_payload)
        if status.status == "succeeded":
            return status

        request = BatchJobCreateRequest.model_validate(input_payload)
        running = status.model_copy(
            update={"status": "running", "updated_at": _utc_now(), "error": None}
        )
        self.store.put_json(_job_key(job_id), running.model_dump(mode="json"))

        predictor = self.predictor_factory()
        items: list[dict[str, Any]] = []

        try:
            for start in range(0, len(request.examples), self.chunk_size):
                chunk: list[BatchPredictExample] = request.examples[
                    start : start + self.chunk_size
                ]
                responses = predictor.predict_open_set_batch(
                    examples=chunk,
                    top_k=request.top_k,
                    profile=request.profile,
                )
                for response in responses:
                    items.append(
                        BatchJobItem(
                            head=response.head,
                            tail=response.tail,
                            relation=response.predictions[0],
                            decision=response.decision,
                            accepted=response.accepted,
                            profile=response.profile,
                            threshold=response.threshold,
                        ).model_dump(mode="json")
                    )

                running = running.model_copy(
                    update={
                        "processed": len(items),
                        "updated_at": _utc_now(),
                        "model_id": predictor.model_id,
                    }
                )
                self.store.put_json(_job_key(job_id), running.model_dump(mode="json"))

            self.store.put_json(_result_key(job_id), {"items": items})
            succeeded = running.model_copy(
                update={
                    "status": "succeeded",
                    "processed": len(items),
                    "updated_at": _utc_now(),
                    "model_id": predictor.model_id,
                }
            )
            self.store.put_json(_job_key(job_id), succeeded.model_dump(mode="json"))
            return succeeded
        except Exception as exc:
            failed = running.model_copy(
                update={
                    "status": "failed",
                    "updated_at": _utc_now(),
                    "error": str(exc)[:1000],
                }
            )
            self.store.put_json(_job_key(job_id), failed.model_dump(mode="json"))
            raise
