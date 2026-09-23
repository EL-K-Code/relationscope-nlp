"""Export consented RelationScope feedback as JSONL for retraining/review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from relationscope.ops.store import get_operations_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--ops-bucket", default=None)
    args = parser.parse_args()

    if args.ops_bucket:
        os.environ["RELATIONSCOPE_OPS_BUCKET"] = args.ops_bucket

    store = get_operations_store()
    rows = store.list_json("feedback", limit=args.limit)

    exported = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in reversed(rows):
            if not row.get("consent_to_store_text"):
                skipped += 1
                continue
            if "text" not in row or "head" not in row or "tail" not in row:
                skipped += 1
                continue
            verdict = row.get("verdict")
            prediction = row.get("prediction", {})
            relation_id = (
                row.get("corrected_relation_id")
                if verdict == "correct"
                else prediction.get("relation_id")
                if verdict == "confirm"
                else None
            )
            relation_name = (
                row.get("corrected_relation_name")
                if verdict == "correct"
                else prediction.get("relation_name")
                if verdict == "confirm"
                else "UNKNOWN_RELATION"
                if verdict == "unknown"
                else None
            )
            if verdict == "skip" or relation_name is None:
                skipped += 1
                continue

            record = {
                "feedback_id": row["feedback_id"],
                "text": row["text"],
                "head": row["head"],
                "tail": row["tail"],
                "relation_id": relation_id,
                "relation_name": relation_name,
                "verdict": verdict,
                "source_model_id": prediction.get("model_id"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                "exported": exported,
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
