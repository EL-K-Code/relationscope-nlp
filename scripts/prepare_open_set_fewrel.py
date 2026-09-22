"""Materialize the strict relation-level FewRel open-set protocol."""

from __future__ import annotations

import json
from pathlib import Path

from relationscope.data.io import write_jsonl
from relationscope.data.open_set import load_open_set_protocol

RAW = Path("data/raw/fewrel")
OUT = Path("data/processed/fewrel_open_set")


def main() -> None:
    protocol = load_open_set_protocol(RAW, seed=42)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train_known.jsonl", protocol.train_known)
    write_jsonl(OUT / "validation_known.jsonl", protocol.validation_known)
    write_jsonl(OUT / "test_known.jsonl", protocol.test_known)
    write_jsonl(OUT / "calibration_unknown.jsonl", protocol.calibration_unknown)
    write_jsonl(OUT / "test_unknown.jsonl", protocol.test_unknown)

    report = {
        "protocol": "fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown",
        "seed": 42,
        "known_relation_count": len(protocol.known_relations),
        "calibration_unknown_relation_count": len(
            protocol.calibration_unknown_relations
        ),
        "test_unknown_relation_count": len(protocol.test_unknown_relations),
        "known_relations": list(protocol.known_relations),
        "calibration_unknown_relations": list(
            protocol.calibration_unknown_relations
        ),
        "test_unknown_relations": list(protocol.test_unknown_relations),
        "split_sizes": {
            "train_known": len(protocol.train_known),
            "validation_known": len(protocol.validation_known),
            "test_known": len(protocol.test_known),
            "calibration_unknown": len(protocol.calibration_unknown),
            "test_unknown": len(protocol.test_unknown),
        },
        "leakage_guardrail": (
            "Official validation relations are used only as final unseen-relation test data."
        ),
    }
    (OUT / "protocol.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
