import pytest

from relationscope.constants import HEAD_END, HEAD_START, TAIL_END, TAIL_START
from relationscope.data.markers import mark_character_spans, mark_fewrel_tokens
from relationscope.data.schema import EntitySpan


def test_mark_fewrel_tokens_handles_multi_token_entities() -> None:
    tokens = ["Marie", "Curie", "was", "born", "in", "Warsaw", "."]
    marked = mark_fewrel_tokens(tokens, [[0, 1]], [[5]])

    assert marked == [
        HEAD_START,
        "Marie",
        "Curie",
        HEAD_END,
        "was",
        "born",
        "in",
        TAIL_START,
        "Warsaw",
        TAIL_END,
        ".",
    ]


def test_mark_character_spans() -> None:
    text = "Marie Curie was born in Warsaw."
    head = EntitySpan(text="Marie Curie", start=0, end=11)
    tail = EntitySpan(text="Warsaw", start=24, end=30)

    marked = mark_character_spans(text, head, tail)

    assert marked == "[HEAD] Marie Curie [/HEAD] was born in [TAIL] Warsaw [/TAIL]."


def test_character_span_rejects_wrong_offsets() -> None:
    text = "Marie Curie was born in Warsaw."
    head = EntitySpan(text="Marie", start=1, end=6)
    tail = EntitySpan(text="Warsaw", start=24, end=30)

    with pytest.raises(ValueError, match="do not match"):
        mark_character_spans(text, head, tail)
