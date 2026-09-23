"""Promote one immutable GCS model release to the production alias."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from google.cloud import storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--releases-prefix", default="models/releases")
    parser.add_argument("--target-prefix", default="models/current")
    args = parser.parse_args()

    source_prefix = (
        f"{args.releases_prefix.strip('/')}/{args.model_version}".rstrip("/")
    )
    target_prefix = args.target_prefix.strip("/")

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    source_blobs = [
        blob
        for blob in client.list_blobs(args.bucket, prefix=f"{source_prefix}/")
        if not blob.name.endswith("/")
    ]
    if not source_blobs:
        raise RuntimeError(
            f"registered release not found: gs://{args.bucket}/{source_prefix}"
        )

    manifest_blob = bucket.blob(f"{source_prefix}/manifest.json")
    if not manifest_blob.exists():
        raise RuntimeError("registered release has no manifest.json")

    manifest = json.loads(manifest_blob.download_as_text())
    if manifest.get("model_version") != args.model_version:
        raise RuntimeError("manifest model_version does not match requested release")

    for existing in client.list_blobs(args.bucket, prefix=f"{target_prefix}/"):
        if not existing.name.endswith("/"):
            existing.delete()

    for source in source_blobs:
        relative = source.name[len(source_prefix) + 1 :]
        bucket.copy_blob(source, bucket, f"{target_prefix}/{relative}")

    promotion_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    promotion = {
        "promotion_id": promotion_id,
        "promoted_at": datetime.now(UTC).isoformat(),
        "model_version": args.model_version,
        "source": f"gs://{args.bucket}/{source_prefix}",
        "target": f"gs://{args.bucket}/{target_prefix}",
    }
    bucket.blob(
        f"models/promotions/{promotion_id}.json"
    ).upload_from_string(
        json.dumps(promotion, sort_keys=True),
        content_type="application/json",
    )

    print(json.dumps(promotion, indent=2))


if __name__ == "__main__":
    main()
