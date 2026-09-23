"""Register an immutable RelationScope model release in GCS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from google.cloud import storage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(bundle_dir: Path) -> list[Path]:
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
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--prefix", default="models/releases")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = _collect(args.bundle_dir)
    version_prefix = f"{args.prefix.strip('/')}/{args.model_version}"
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    existing = list(client.list_blobs(args.bucket, prefix=f"{version_prefix}/"))
    if existing and not args.force:
        raise RuntimeError(
            f"release gs://{args.bucket}/{version_prefix} already exists; "
            "immutable releases require a new model version"
        )
    if existing and args.force:
        for blob in existing:
            blob.delete()

    manifest = {
        "model_version": args.model_version,
        "registered_as": f"gs://{args.bucket}/{version_prefix}",
        "files": {
            str(path.relative_to(args.bundle_dir)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }

    manifest_path = args.bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files.append(manifest_path)

    for path in files:
        relative = str(path.relative_to(args.bundle_dir)).replace("\\", "/")
        object_name = f"{version_prefix}/{relative}"
        bucket.blob(object_name).upload_from_filename(path)
        print(f"uploaded {relative}")

    print(
        json.dumps(
            {
                "model_version": args.model_version,
                "release_uri": f"gs://{args.bucket}/{version_prefix}",
                "files": len(files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
