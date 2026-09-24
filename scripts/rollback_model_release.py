"""Rollback the promoted model alias to a previous immutable release."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from google.cloud import storage


def _promotion_history(
    client: storage.Client,
    bucket_name: str,
) -> list[tuple[str, dict[str, object]]]:
    records: list[tuple[str, dict[str, object]]] = []
    for blob in client.list_blobs(bucket_name, prefix="models/promotions/"):
        if blob.name.endswith("/"):
            continue
        try:
            payload = json.loads(blob.download_as_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append((blob.name, payload))
    records.sort(key=lambda item: item[0], reverse=True)
    return records


def _copy_release(
    client: storage.Client,
    bucket_name: str,
    *,
    model_version: str,
    releases_prefix: str,
    target_prefix: str,
) -> None:
    bucket = client.bucket(bucket_name)
    source_prefix = f"{releases_prefix.strip('/')}/{model_version}"
    source_blobs = [
        blob
        for blob in client.list_blobs(bucket_name, prefix=f"{source_prefix}/")
        if not blob.name.endswith("/")
    ]
    if not source_blobs:
        raise RuntimeError(
            f"release not found: gs://{bucket_name}/{source_prefix}"
        )

    manifest = bucket.blob(f"{source_prefix}/manifest.json")
    if not manifest.exists():
        raise RuntimeError("rollback target has no manifest.json")

    for existing in client.list_blobs(
        bucket_name,
        prefix=f"{target_prefix.strip('/')}/",
    ):
        if not existing.name.endswith("/"):
            existing.delete()

    for source in source_blobs:
        relative = source.name[len(source_prefix) + 1 :]
        bucket.copy_blob(
            source,
            bucket,
            f"{target_prefix.strip('/')}/{relative}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--releases-prefix", default="models/releases")
    parser.add_argument("--target-prefix", default="models/current")
    args = parser.parse_args()

    client = storage.Client()
    history = _promotion_history(client, args.bucket)

    target = args.model_version
    if target is None:
        versions = [
            str(payload.get("model_version"))
            for _, payload in history
            if payload.get("model_version")
        ]
        unique: list[str] = []
        for version in versions:
            if version not in unique:
                unique.append(version)
        if len(unique) < 2:
            raise RuntimeError(
                "automatic rollback requires at least two distinct promotion records"
            )
        target = unique[1]

    _copy_release(
        client,
        args.bucket,
        model_version=target,
        releases_prefix=args.releases_prefix,
        target_prefix=args.target_prefix,
    )

    now = datetime.now(UTC)
    rollback_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    payload = {
        "rollback_id": rollback_id,
        "rolled_back_at": now.isoformat(),
        "model_version": target,
        "target": f"gs://{args.bucket}/{args.target_prefix.strip('/')}",
    }
    client.bucket(args.bucket).blob(
        f"models/rollbacks/{rollback_id}.json"
    ).upload_from_string(
        json.dumps(payload, sort_keys=True),
        content_type="application/json",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
