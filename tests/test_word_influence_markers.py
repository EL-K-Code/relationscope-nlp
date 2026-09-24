from relationscope.inference.predictor import contains_entity_marker_fragment


def test_contains_entity_marker_fragment_rejects_structural_tokens_with_punctuation() -> None:
    assert contains_entity_marker_fragment("[HEAD]")
    assert contains_entity_marker_fragment("[/HEAD]")
    assert contains_entity_marker_fragment("[TAIL]")
    assert contains_entity_marker_fragment("[/TAIL].")
    assert contains_entity_marker_fragment("([TAIL])")


def test_contains_entity_marker_fragment_keeps_normal_words() -> None:
    assert not contains_entity_marker_fragment("Marie")
    assert not contains_entity_marker_fragment("born")
    assert not contains_entity_marker_fragment("Warsaw")
