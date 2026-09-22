"""FewRel ingestion for the RelationScope closed-set benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from relationscope.data.markers import mark_fewrel_tokens
from relationscope.data.schema import RelationRecord


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relation_metadata(pid2name: Mapping[str, list[str]], relation_id: str) -> tuple[str, str]:
    values = pid2name.get(relation_id)
    if not values:
        return relation_id, ""
    name = str(values[0]).strip()
    description = str(values[1]).strip() if len(values) > 1 else ""
    return name, description


def _entity_text(entity: object) -> str:
    if isinstance(entity, list) and entity:
        return str(entity[0])
    raise ValueError(f"invalid FewRel entity: {entity!r}")


def flatten_split(
    raw: Mapping[str, list[dict[str, Any]]],
    pid2name: Mapping[str, list[str]],
    source_split: str,
) -> list[RelationRecord]:
    """Flatten a FewRel relation->examples mapping into typed records."""
    records: list[RelationRecord] = []

    for relation_id, examples in raw.items():
        relation_name, relation_description = _relation_metadata(pid2name, relation_id)
        for example in examples:
            tokens = [str(token) for token in example["tokens"]]
            head = example["h"]
            tail = example["t"]
            marked_tokens = mark_fewrel_tokens(tokens, head[2], tail[2])

            records.append(
                RelationRecord(
                    relation_id=relation_id,
                    relation_name=relation_name,
                    relation_description=relation_description,
                    tokens=tokens,
                    marked_text=" ".join(marked_tokens),
                    head_text=_entity_text(head),
                    tail_text=_entity_text(tail),
                    source_split=source_split,
                )
            )
    return records


def load_closed_set_80(raw_dir: Path) -> list[RelationRecord]:
    """Load 64 train + 16 validation FewRel relations for closed-set classification."""
    train = load_json(raw_dir / "train_wiki.json")
    validation = load_json(raw_dir / "val_wiki.json")
    pid2name = load_json(raw_dir / "pid2name.json")

    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"official train/validation relation sets unexpectedly overlap: {overlap}")

    records = flatten_split(train, pid2name, "official_train")
    records.extend(flatten_split(validation, pid2name, "official_validation"))
    return records
