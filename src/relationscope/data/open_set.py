"""Open-set FewRel protocol.

The protocol uses relation-level holdouts rather than example-level OOD simulation:

- 56 official-train relations -> known classes used for classifier training;
- 8 official-train relations -> calibration-unknown classes used only to tune abstention;
- 16 official-validation relations -> final unseen-relation test set.

This preserves a strict relation-level separation between training and unknown evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from relationscope.data.fewrel import flatten_split, load_json
from relationscope.data.schema import RelationRecord
from relationscope.data.splits import stratified_relation_split


@dataclass(frozen=True)
class OpenSetFewRel:
    train_known: list[RelationRecord]
    validation_known: list[RelationRecord]
    test_known: list[RelationRecord]
    calibration_unknown: list[RelationRecord]
    test_unknown: list[RelationRecord]
    known_relations: tuple[str, ...]
    calibration_unknown_relations: tuple[str, ...]
    test_unknown_relations: tuple[str, ...]


def load_open_set_protocol(
    raw_dir: Path,
    *,
    seed: int = 42,
    calibration_unknown_relation_count: int = 8,
) -> OpenSetFewRel:
    """Create a deterministic relation-level open-set protocol from FewRel 1.0."""
    train_raw = load_json(raw_dir / "train_wiki.json")
    validation_raw = load_json(raw_dir / "val_wiki.json")
    pid2name = load_json(raw_dir / "pid2name.json")

    train_relations = sorted(train_raw)
    validation_relations = sorted(validation_raw)

    if set(train_relations).intersection(validation_relations):
        raise ValueError("official FewRel train and validation relation sets must be disjoint")
    if calibration_unknown_relation_count <= 0:
        raise ValueError("calibration_unknown_relation_count must be positive")
    if calibration_unknown_relation_count >= len(train_relations):
        raise ValueError("calibration unknown set would consume all train relations")

    rng = random.Random(seed)
    shuffled = train_relations[:]
    rng.shuffle(shuffled)

    calibration_unknown_relations = tuple(
        sorted(shuffled[:calibration_unknown_relation_count])
    )
    known_relations = tuple(sorted(shuffled[calibration_unknown_relation_count:]))
    test_unknown_relations = tuple(validation_relations)

    known_raw = {relation: train_raw[relation] for relation in known_relations}
    calibration_raw = {
        relation: train_raw[relation]
        for relation in calibration_unknown_relations
    }

    known_records = flatten_split(known_raw, pid2name, "official_train_known")
    known_splits = stratified_relation_split(
        known_records,
        train_ratio=0.8,
        validation_ratio=0.1,
        seed=seed,
    )

    calibration_unknown = flatten_split(
        calibration_raw,
        pid2name,
        "official_train_calibration_unknown",
    )
    test_unknown = flatten_split(
        validation_raw,
        pid2name,
        "official_validation_test_unknown",
    )

    return OpenSetFewRel(
        train_known=known_splits["train"],
        validation_known=known_splits["validation"],
        test_known=known_splits["test"],
        calibration_unknown=calibration_unknown,
        test_unknown=test_unknown,
        known_relations=known_relations,
        calibration_unknown_relations=calibration_unknown_relations,
        test_unknown_relations=test_unknown_relations,
    )
