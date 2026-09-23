"""Read-only model registry view backed by GCS manifests."""

from __future__ import annotations

import json
import os
from datetime import datetime
from urllib.parse import urlparse

from google.cloud import storage

from relationscope.data.v2_schema import ModelRegistryEntry, ModelRegistryResponse


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"invalid GCS URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


class ModelRegistryService:
    """Discover immutable model releases without loading them into API memory."""

    def __init__(
        self,
        *,
        current_model_id: str,
        model_gcs_uri: str | None = None,
        releases_prefix: str = "models/releases",
    ) -> None:
        self.current_model_id = current_model_id
        self.model_gcs_uri = model_gcs_uri
        self.releases_prefix = releases_prefix.strip("/")

    def list(self) -> ModelRegistryResponse:
        entries: list[ModelRegistryEntry] = []

        if self.model_gcs_uri:
            bucket_name, _ = _parse_gs_uri(self.model_gcs_uri)
            client = storage.Client()
            for blob in client.list_blobs(
                bucket_name,
                prefix=f"{self.releases_prefix}/",
            ):
                if not blob.name.endswith("/manifest.json"):
                    continue
                try:
                    manifest = json.loads(blob.download_as_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                version = manifest.get("model_version")
                if not isinstance(version, str) or not version.strip():
                    continue

                release_prefix = blob.name[: -len("/manifest.json")]
                created_at: datetime | None = blob.time_created
                entries.append(
                    ModelRegistryEntry(
                        model_version=version.strip(),
                        gcs_prefix=f"gs://{bucket_name}/{release_prefix}",
                        promoted=version.strip() == self.current_model_id,
                        created_at=created_at,
                    )
                )

        if not any(item.model_version == self.current_model_id for item in entries):
            entries.append(
                ModelRegistryEntry(
                    model_version=self.current_model_id,
                    gcs_prefix=self.model_gcs_uri or "local",
                    promoted=True,
                )
            )

        entries.sort(
            key=lambda item: item.created_at.isoformat() if item.created_at else "",
            reverse=True,
        )
        return ModelRegistryResponse(
            current_model_id=self.current_model_id,
            entries=entries,
        )


def registry_from_env(current_model_id: str) -> ModelRegistryService:
    return ModelRegistryService(
        current_model_id=current_model_id,
        model_gcs_uri=os.getenv("RELATIONSCOPE_MODEL_GCS_URI"),
        releases_prefix=os.getenv(
            "RELATIONSCOPE_MODEL_RELEASES_PREFIX",
            "models/releases",
        ),
    )
