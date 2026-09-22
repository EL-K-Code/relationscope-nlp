"""Model-bundle resolution and verified download from Google Cloud Storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

_DOWNLOAD_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"invalid GCS URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _is_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "relations.json").is_file()
        and (path / "classifier.pt").is_file()
        and (path / "encoder").is_dir()
        and (path / "tokenizer").is_dir()
    )


def _safe_child(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe model artifact path: {relative_name!r}")

    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise RuntimeError(f"model artifact escapes destination: {relative_name!r}")
    return candidate


def _verify_manifest(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise RuntimeError("invalid model manifest: files must be an object")

    for relative_name, expected_sha in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_sha, str):
            raise RuntimeError("invalid model manifest entry")

        candidate = _safe_child(path, relative_name)
        if not candidate.is_file():
            raise RuntimeError(f"model artifact missing after download: {relative_name}")

        actual_sha = _sha256(candidate)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"model artifact checksum mismatch for {relative_name}: "
                f"{actual_sha} != {expected_sha}"
            )


def download_model_bundle(gcs_uri: str, destination: Path) -> Path:
    """Download one model bundle atomically from a GCS prefix."""
    bucket_name, prefix = _parse_gs_uri(gcs_uri)
    client = storage.Client()

    temp = destination.with_name(destination.name + ".downloading")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)

    try:
        matched = 0
        normalized_prefix = f"{prefix}/" if prefix else ""
        for blob in client.list_blobs(bucket_name, prefix=normalized_prefix):
            if blob.name.endswith("/"):
                continue

            relative = blob.name[len(normalized_prefix) :]
            if not relative:
                continue

            local_path = _safe_child(temp, relative)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
            matched += 1

        if matched == 0:
            raise RuntimeError(f"no model artifacts found at {gcs_uri}")

        _verify_manifest(temp)

        if not _is_ready(temp):
            raise RuntimeError(
                f"downloaded bundle from {gcs_uri} is incomplete or incompatible"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        temp.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def resolve_model_bundle() -> Path:
    """Return a verified local model bundle, downloading from GCS if configured."""
    destination = Path(
        os.getenv(
            "RELATIONSCOPE_MODEL_DIR",
            "/tmp/relationscope/model",
        )
    )

    if _is_ready(destination):
        _verify_manifest(destination)
        return destination

    gcs_uri = os.getenv("RELATIONSCOPE_MODEL_GCS_URI")
    if not gcs_uri:
        raise RuntimeError(
            f"no complete model bundle at {destination} and "
            "RELATIONSCOPE_MODEL_GCS_URI is not configured"
        )

    with _DOWNLOAD_LOCK:
        if _is_ready(destination):
            _verify_manifest(destination)
            return destination
        return download_model_bundle(gcs_uri, destination)
