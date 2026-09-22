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
        path.exists()
        and (path / "relations.json").exists()
        and (path / "classifier.pt").exists()
        and (path / "encoder").is_dir()
        and (path / "tokenizer").is_dir()
    )


def _verify_manifest(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise RuntimeError("invalid model manifest: files must be an object")

    for relative_name, expected_sha in files.items():
        candidate = path / relative_name
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
    bucket = client.bucket(bucket_name)

    temp = destination.with_name(destination.name + ".downloading")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)

    matched = 0
    normalized_prefix = f"{prefix}/" if prefix else ""
    for blob in client.list_blobs(bucket_name, prefix=normalized_prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(normalized_prefix) :]
        if not relative:
            continue
        local_path = temp / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        matched += 1

    if matched == 0:
        shutil.rmtree(temp, ignore_errors=True)
        raise RuntimeError(f"no model artifacts found at {gcs_uri}")

    _verify_manifest(temp)

    if destination.exists():
        shutil.rmtree(destination)
    temp.rename(destination)

    if not _is_ready(destination):
        raise RuntimeError(
            f"downloaded bundle at {destination} is incomplete or incompatible"
        )
    return destination


def resolve_model_bundle() -> Path:
    """Return a usable local model bundle, downloading it from GCS when configured."""
    destination = Path(
        os.getenv(
            "RELATIONSCOPE_MODEL_DIR",
            "/tmp/relationscope/model",
        )
    )

    if _is_ready(destination):
        return destination

    gcs_uri = os.getenv("RELATIONSCOPE_MODEL_GCS_URI")
    if not gcs_uri:
        return destination

    with _DOWNLOAD_LOCK:
        if _is_ready(destination):
            return destination
        return download_model_bundle(gcs_uri, destination)
