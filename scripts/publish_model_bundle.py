"""Publish a trained RelationScope model bundle to Google Cloud Storage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"invalid GCS URI: {uri!r}")

    prefix = parsed.path.lstrip("/").rstrip("/")
    if not prefix:
        raise ValueError("model destination must include a non-empty GCS prefix")
    return parsed.netloc, prefix


def collect_files(bundle_dir: Path) -> list[Path]:
    required = [
        bundle_dir / "relations.json",
        bundle_dir / "classifier.pt",
        bundle_dir / "encoder",
        bundle_dir / "tokenizer",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("incomplete model bundle: " + ", ".join(missing))

    return sorted(
        path
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def clear_prefix(client: storage.Client, bucket_name: str, prefix: str) -> int:
    """Delete current objects under a model prefix before publishing replacement files."""
    normalized_prefix = f"{prefix}/"
    deleted = 0
    bucket = client.bucket(bucket_name)

    for blob in client.list_blobs(bucket_name, prefix=normalized_prefix):
        if blob.name.endswith("/"):
            continue
        bucket.blob(blob.name).delete()
        deleted += 1

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("destination", help="gs://bucket/non-empty-prefix")
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    files = collect_files(args.bundle_dir)
    manifest = {
        "model_version": args.model_version or args.bundle_dir.name,
        "files": {
            str(path.relative_to(args.bundle_dir)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }

    local_manifest = args.bundle_dir / "manifest.json"
    local_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    files.append(local_manifest)

    bucket_name, prefix = parse_gs_uri(args.destination)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    deleted = clear_prefix(client, bucket_name, prefix)
    print(f"cleared {deleted} existing objects under gs://{bucket_name}/{prefix}/")

    for path in files:
        relative = str(path.relative_to(args.bundle_dir)).replace("\\", "/")
        object_name = "/".join((prefix, relative))
        print(f"upload {relative} -> gs://{bucket_name}/{object_name}")
        bucket.blob(object_name).upload_from_filename(path)

    print(
        json.dumps(
            {
                "destination": args.destination,
                "files": len(files),
                "deleted_previous_objects": deleted,
                "model_version": manifest["model_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
