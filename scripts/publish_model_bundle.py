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
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("destination", help="gs://bucket/prefix")
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

    for path in files:
        relative = str(path.relative_to(args.bundle_dir)).replace("\\", "/")
        object_name = "/".join(part for part in (prefix, relative) if part)
        print(f"upload {relative} -> gs://{bucket_name}/{object_name}")
        bucket.blob(object_name).upload_from_filename(path)

    print(
        json.dumps(
            {
                "destination": args.destination,
                "files": len(files),
                "model_version": manifest["model_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
