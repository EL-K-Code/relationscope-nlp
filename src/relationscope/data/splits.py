"""Deterministic, per-relation dataset splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from relationscope.data.schema import RelationRecord


def stratified_relation_split(
    records: Iterable[RelationRecord],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[RelationRecord]]:
    """Split independently inside each relation to avoid class imbalance by construction."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train and validation ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    grouped: dict[str, list[RelationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.relation_id].append(record)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}

    for relation_id in sorted(grouped):
        items = grouped[relation_id][:]
        rng.shuffle(items)
        count = len(items)

        train_end = int(count * train_ratio)
        validation_end = train_end + int(count * validation_ratio)

        if train_end == 0 or validation_end == train_end or validation_end >= count:
            raise ValueError(f"relation {relation_id} has too few examples for requested split")

        splits["train"].extend(items[:train_end])
        splits["validation"].extend(items[train_end:validation_end])
        splits["test"].extend(items[validation_end:])

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits
