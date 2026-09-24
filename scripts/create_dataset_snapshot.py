"""Create an immutable V3 reviewed-data snapshot from production operations storage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from relationscope.ops.store import get_operations_store
from relationscope.training.dataset_versioning import DatasetSnapshotBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--ops-bucket", default=None)
    parser.add_argument("--limit-per-source", type=int, default=10_000)
    args = parser.parse_args()

    if args.ops_bucket:
        os.environ["RELATIONSCOPE_OPS_BUCKET"] = args.ops_bucket

    builder = DatasetSnapshotBuilder(get_operations_store())
    jsonl_path, manifest_path, manifest = builder.write_snapshot(
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        limit_per_source=args.limit_per_source,
    )
    print(
        json.dumps(
            {
                "dataset_version": manifest.dataset_version,
                "records": manifest.record_count,
                "sha256": manifest.sha256,
                "examples": str(jsonl_path),
                "manifest": str(manifest_path),
                "source_counts": manifest.source_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
