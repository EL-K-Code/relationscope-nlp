from __future__ import annotations

from relationscope.document.entity_extractor import HeuristicEntityExtractor
from relationscope.document.pdf_text import normalize_pdf_pages


def test_pdf_text_normalization_joins_only_visual_wraps() -> None:
    raw = (
        "RelationScope V3 - PDF Smoke Test\n"
        "Synthetic document for validating PDF extraction, entity detection, relation generation,\n"
        "open-set inference, and knowledge-graph assembly.\n"
        "Test text\n"
        "Albert Einstein worked at Princeton University.\n"
        "Candidate generation\n"
        "Entity pairs should be generated only within the same sentence.\n"
    )

    normalized = normalize_pdf_pages([raw])

    assert (
        "relation generation, open-set inference, and knowledge-graph assembly."
        in normalized
    )
    assert "RelationScope V3 - PDF Smoke Test\nSynthetic document" in normalized
    assert "Test text\nAlbert Einstein worked at Princeton University." in normalized
    assert "Candidate generation\nEntity pairs should" in normalized


def test_pdf_text_normalization_dehyphenates_visual_line_breaks() -> None:
    normalized = normalize_pdf_pages(
        ["The organiza-\ntion worked with Princeton University."]
    )
    assert normalized == "The organization worked with Princeton University."


def test_fallback_ner_rejects_layout_words_and_connector_prefixes() -> None:
    text = (
        "Entity detection. "
        "Active learning. "
        "Review candidates may be identified. "
        "OpenAI is based in San Francisco."
    )
    entities = HeuristicEntityExtractor().extract(text)

    assert [item.text for item in entities] == ["OpenAI", "San Francisco"]
    assert all("Entity de" not in item.text for item in entities)
    assert all("Active le" not in item.text for item in entities)
