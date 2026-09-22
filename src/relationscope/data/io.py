"""JSONL serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from relationscope.data.schema import RelationRecord


def write_jsonl(path: Path, records: list[RelationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[RelationRecord]:
    records: list[RelationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RelationRecord.model_validate_json(line))
    return records
