"""Version reviewed production examples into immutable training snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from relationscope.data.schema import EntitySpan
from relationscope.data.v3_schema import DatasetSnapshotManifest, TrainingCandidate
from relationscope.ops.store import OperationsStore


def _canonical_record(record: TrainingCandidate) -> str:
    return json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class DatasetSnapshotBuilder:
    """Build deterministic JSONL snapshots from explicitly reviewable production data."""

    def __init__(self, store: OperationsStore) -> None:
        self.store = store

    def _feedback_candidates(self, limit: int) -> list[TrainingCandidate]:
        candidates: list[TrainingCandidate] = []
        for row in self.store.list_json("feedback", limit=limit):
            if not row.get("consent_to_store_text"):
                continue
            if not all(key in row for key in ("text", "head", "tail", "feedback_id")):
                continue

            verdict = row.get("verdict")
            prediction = row.get("prediction", {})
            if not isinstance(prediction, dict):
                continue

            if verdict == "confirm":
                relation_id = prediction.get("relation_id")
                relation_name = prediction.get("relation_name")
            elif verdict == "correct":
                relation_id = row.get("corrected_relation_id")
                relation_name = row.get("corrected_relation_name")
            elif verdict == "unknown":
                relation_id = None
                relation_name = "UNKNOWN_RELATION"
            else:
                continue

            if not isinstance(relation_name, str) or not relation_name:
                continue

            candidates.append(
                TrainingCandidate(
                    text=str(row["text"]),
                    head=EntitySpan.model_validate(row["head"]),
                    tail=EntitySpan.model_validate(row["tail"]),
                    relation_id=str(relation_id) if relation_id is not None else None,
                    relation_name=relation_name,
                    source="feedback",
                    source_id=str(row["feedback_id"]),
                )
            )
        return candidates

    def _review_candidates(self, limit: int) -> list[TrainingCandidate]:
        candidates: list[TrainingCandidate] = []
        for row in self.store.list_json("review/queue", limit=limit):
            if row.get("status") != "reviewed":
                continue
            verdict = row.get("reviewer_verdict")
            if verdict not in {"confirm", "correct", "unknown"}:
                continue
            if not all(key in row for key in ("text", "head", "tail", "review_id")):
                continue

            relation_name = row.get("reviewed_relation_name")
            if not isinstance(relation_name, str) or not relation_name:
                continue

            relation_id = row.get("reviewed_relation_id")
            candidates.append(
                TrainingCandidate(
                    text=str(row["text"]),
                    head=EntitySpan.model_validate(row["head"]),
                    tail=EntitySpan.model_validate(row["tail"]),
                    relation_id=str(relation_id) if relation_id is not None else None,
                    relation_name=relation_name,
                    source="active_learning",
                    source_id=str(row["review_id"]),
                )
            )
        return candidates

    def collect(self, *, limit_per_source: int = 10_000) -> list[TrainingCandidate]:
        raw = self._feedback_candidates(limit_per_source)
        raw.extend(self._review_candidates(limit_per_source))

        deduplicated: dict[str, TrainingCandidate] = {}
        for candidate in raw:
            identity_payload: dict[str, Any] = {
                "text": candidate.text,
                "head": candidate.head.model_dump(mode="json"),
                "tail": candidate.tail.model_dump(mode="json"),
                "relation_id": candidate.relation_id,
                "relation_name": candidate.relation_name,
            }
            identity = hashlib.sha256(
                json.dumps(
                    identity_payload,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            deduplicated.setdefault(identity, candidate)

        return sorted(
            deduplicated.values(),
            key=lambda item: (
                item.relation_name,
                item.text,
                item.head.start,
                item.tail.start,
            ),
        )

    def write_snapshot(
        self,
        *,
        output_dir: Path,
        dataset_version: str,
        limit_per_source: int = 10_000,
    ) -> tuple[Path, Path, DatasetSnapshotManifest]:
        records = self.collect(limit_per_source=limit_per_source)
        if not dataset_version.strip():
            raise ValueError("dataset_version must not be empty")

        snapshot_dir = output_dir / dataset_version
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        jsonl_path = snapshot_dir / "examples.jsonl"

        digest = hashlib.sha256()
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                line = _canonical_record(record)
                handle.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))

        source_counts = Counter(record.source for record in records)
        relation_counts = Counter(record.relation_name for record in records)
        manifest = DatasetSnapshotManifest(
            dataset_version=dataset_version,
            created_at=datetime.now(UTC),
            record_count=len(records),
            sha256=digest.hexdigest(),
            source_counts=dict(sorted(source_counts.items())),
            relation_counts=dict(sorted(relation_counts.items())),
        )
        manifest_path = snapshot_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return jsonl_path, manifest_path, manifest
