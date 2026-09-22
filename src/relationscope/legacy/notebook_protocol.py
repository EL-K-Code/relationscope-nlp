"""Utilities that reproduce the ENSAI notebook's experimental protocol.

This module intentionally preserves the original educational choices:
- seven sampled relations for the first experiments;
- global shuffle with seed 42;
- 70/30 train/test split, then 80/20 train/validation split;
- vocabulary built only from the training subset;
- lowercase tokenization;
- top-5000 input vocabulary.

The production benchmark lives elsewhere and should not be confused with this lineage.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable

from relationscope.data.schema import RelationRecord

NOTEBOOK_SEED = 42

NOTEBOOK_RELATIONS = (
    "P137",   # operator
    "P105",   # taxon rank
    "P26",    # spouse
    "P2094",  # competition class
    "P1877",  # after a work by
    "P1411",  # nominated for
    "P136",   # genre
)

SCALING_K_VALUES = (7, 10, 15, 20, 30)


def notebook_label(record: RelationRecord) -> str:
    """Recreate the notebook label: relation name + Wikidata description."""
    if record.relation_description:
        return f"{record.relation_name}, {record.relation_description}"
    return record.relation_name


def select_records(
    records: Iterable[RelationRecord],
    relation_ids: Iterable[str],
) -> list[RelationRecord]:
    selected = set(relation_ids)
    return [record for record in records if record.relation_id in selected]


def notebook_split(
    records: Iterable[RelationRecord],
    relation_ids: Iterable[str],
    *,
    seed: int = NOTEBOOK_SEED,
) -> dict[str, list[RelationRecord]]:
    """Reproduce the notebook's shuffle and 56/14/30 effective split."""
    data = select_records(records, relation_ids)
    rng = random.Random(seed)
    rng.shuffle(data)

    train_test_cut = int(0.7 * len(data))
    train_pool = data[:train_test_cut]
    test = data[train_test_cut:]

    train_validation_cut = int(0.8 * len(train_pool))
    train = train_pool[:train_validation_cut]
    validation = train_pool[train_validation_cut:]

    return {"train": train, "validation": validation, "test": test}


def notebook_input_vocabulary(
    train_records: Iterable[RelationRecord],
    *,
    max_tokens: int = 5000,
) -> dict[str, int]:
    """Build the notebook's lowercase top-5000 vocabulary."""
    counter: Counter[str] = Counter()
    for record in train_records:
        counter.update(token.lower() for token in record.marked_text.split())

    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(max_tokens):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_marked_text(
    text: str,
    vocabulary: dict[str, int],
    *,
    max_length: int,
) -> list[int]:
    """Lowercase, integer-encode, truncate and right-pad as in the notebook."""
    tokens = text.lower().split()
    ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokens]
    ids = ids[:max_length]
    return ids + [vocabulary["<pad>"]] * (max_length - len(ids))


def top_k_relations(records: Iterable[RelationRecord], k: int) -> list[str]:
    """Reproduce the notebook's top-K relation selection used in the scaling study."""
    counts = Counter(record.relation_id for record in records)
    return [relation_id for relation_id, _ in counts.most_common(k)]
