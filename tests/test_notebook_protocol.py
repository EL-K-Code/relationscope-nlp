from relationscope.data.schema import RelationRecord
from relationscope.legacy.notebook_protocol import (
    NOTEBOOK_RELATIONS,
    notebook_split,
)


def _record(relation_id: str, index: int) -> RelationRecord:
    return RelationRecord(
        relation_id=relation_id,
        relation_name=relation_id,
        relation_description="description",
        tokens=["a", "b"],
        marked_text=f"[HEAD] a [/HEAD] [TAIL] b [/TAIL] {index}",
        head_text="a",
        tail_text="b",
        source_split="synthetic",
    )


def test_notebook_seven_relation_split_reproduces_expected_sizes() -> None:
    records = [
        _record(relation_id, index)
        for relation_id in NOTEBOOK_RELATIONS
        for index in range(700)
    ]

    splits = notebook_split(records, NOTEBOOK_RELATIONS, seed=42)

    assert len(splits["train"]) == 2744
    assert len(splits["validation"]) == 686
    assert len(splits["test"]) == 1470
