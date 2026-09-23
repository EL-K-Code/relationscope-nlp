"""Durable JSON object storage for RelationScope operational data."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol

from google.cloud import storage


class OperationsStore(Protocol):
    """Minimal object-store contract used by telemetry, feedback and batch jobs."""

    def put_json(self, key: str, payload: dict[str, Any]) -> None: ...

    def get_json(self, key: str) -> dict[str, Any] | None: ...

    def list_json(self, prefix: str, *, limit: int = 500) -> list[dict[str, Any]]: ...


def _safe_local_path(root: Path, key: str) -> Path:
    relative = Path(key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe operations key: {key!r}")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"operations key escapes root: {key!r}")
    return candidate


class LocalOperationsStore:
    """Filesystem implementation for local development and tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        path = _safe_local_path(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        with self._lock:
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)

    def get_json(self, key: str) -> dict[str, Any] | None:
        path = _safe_local_path(self.root, key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid JSON object at {key}")
        return payload

    def list_json(self, prefix: str, *, limit: int = 500) -> list[dict[str, Any]]:
        base = _safe_local_path(self.root, prefix)
        if not base.exists():
            return []

        files = sorted(
            (path for path in base.rglob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        items: list[dict[str, Any]] = []
        for path in files[:limit]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                items.append(payload)
        return items


class GCSOperationsStore:
    """GCS-backed JSON object store. Each event/job is its own object."""

    def __init__(self, bucket_name: str, *, prefix: str = "v2") -> None:
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.strip("/")

    def _name(self, key: str) -> str:
        normalized = key.lstrip("/")
        if ".." in Path(normalized).parts:
            raise ValueError(f"unsafe operations key: {key!r}")
        return f"{self.prefix}/{normalized}" if self.prefix else normalized

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.bucket.blob(self._name(key)).upload_from_string(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            content_type="application/json",
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        blob = self.bucket.blob(self._name(key))
        if not blob.exists():
            return None
        payload = json.loads(blob.download_as_text())
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid JSON object at {key}")
        return payload

    def list_json(self, prefix: str, *, limit: int = 500) -> list[dict[str, Any]]:
        normalized = self._name(prefix).rstrip("/") + "/"
        blobs = [
            blob
            for blob in self.client.list_blobs(self.bucket.name, prefix=normalized)
            if not blob.name.endswith("/")
        ]
        blobs.sort(key=lambda blob: blob.updated or blob.time_created, reverse=True)

        items: list[dict[str, Any]] = []
        for blob in blobs[:limit]:
            payload = json.loads(blob.download_as_text())
            if isinstance(payload, dict):
                items.append(payload)
        return items


_store: OperationsStore | None = None
_store_lock = threading.Lock()


def get_operations_store() -> OperationsStore:
    """Return a process-wide operations store selected from environment variables."""
    global _store
    if _store is not None:
        return _store

    with _store_lock:
        if _store is not None:
            return _store

        bucket = os.getenv("RELATIONSCOPE_OPS_BUCKET")
        if bucket:
            _store = GCSOperationsStore(
                bucket,
                prefix=os.getenv("RELATIONSCOPE_OPS_PREFIX", "v2"),
            )
        else:
            root = Path(os.getenv("RELATIONSCOPE_OPS_DIR", "artifacts/ops"))
            _store = LocalOperationsStore(root)

    return _store


def reset_operations_store_for_tests() -> None:
    """Reset the singleton. Intended for tests only."""
    global _store
    with _store_lock:
        _store = None
