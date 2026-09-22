"""Build RelationScope's deterministic FewRel Closed-Set-80 dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from relationscope.data.fewrel import load_closed_set_80
from relationscope.data.io import write_jsonl
from relationscope.data.splits import stratified_relation_split

RAW = Path("data/raw/fewrel")
OUT = Path("data/processed/fewrel_closed_set_80")


def main() -> None:
    records = load_closed_set_80(RAW)
    splits = stratified_relation_split(records, seed=42)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        write_jsonl(OUT / f"{name}.jsonl", examples)

    relation_meta = {
        record.relation_id: {
            "relation_name": record.relation_name,
            "relation_description": record.relation_description,
        }
        for record in records
    }
    (OUT / "relations.json").write_text(
        json.dumps(dict(sorted(relation_meta.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "protocol": "fewrel_closed_set_80",
        "seed": 42,
        "total_examples": len(records),
        "num_relations": len(relation_meta),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "source_split_counts": dict(Counter(record.source_split for record in records)),
    }
    (OUT / "dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
