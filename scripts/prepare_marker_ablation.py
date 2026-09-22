"""Prepare the three entity-information conditions used by the marker ablation.

The actual Transformer training command is intentionally left to the common trainer.
This script freezes comparable text views so every condition sees the same examples.
"""

from __future__ import annotations

import json
from pathlib import Path

from relationscope.data.io import read_jsonl

DATA = Path("data/processed/fewrel_closed_set_80")
OUT = Path("data/processed/entity_marker_ablation")


def raw_text(record) -> str:
    return " ".join(record.tokens)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "protocol": "entity_marker_ablation",
        "conditions": {
            "A_raw_cls": "raw sentence, CLS-only classifier",
            "B_marked_cls": "[HEAD]/[TAIL] markers, CLS-only classifier",
            "C_marked_entity_pair": (
                "[HEAD]/[TAIL] markers + CLS/HEAD/TAIL/|HEAD-TAIL| pooling"
            ),
        },
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        records = read_jsonl(DATA / f"{split}.jsonl")
        rows = [
            {
                "relation_id": record.relation_id,
                "raw_text": raw_text(record),
                "marked_text": record.marked_text,
                "head_text": record.head_text,
                "tail_text": record.tail_text,
            }
            for record in records
        ]
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = len(rows)

    (OUT / "protocol.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
