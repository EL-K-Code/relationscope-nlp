from collections import Counter

from relationscope.data.schema import RelationRecord
from relationscope.data.splits import stratified_relation_split


def make_record(relation_id: str, index: int) -> RelationRecord:
    return RelationRecord(
        relation_id=relation_id,
        relation_name=relation_id,
        relation_description="",
        tokens=[str(index)],
        marked_text=f"[HEAD] h [/HEAD] x [TAIL] t [/TAIL] {index}",
        head_text="h",
        tail_text="t",
        source_split="test",
    )


def test_stratified_split_preserves_each_relation() -> None:
    records = [
        make_record(relation_id, index)
        for relation_id in ("P1", "P2")
        for index in range(100)
    ]

    splits = stratified_relation_split(records, seed=42)

    assert len(splits["train"]) == 160
    assert len(splits["validation"]) == 20
    assert len(splits["test"]) == 20

    for split in splits.values():
        counts = Counter(record.relation_id for record in split)
        assert set(counts) == {"P1", "P2"}
