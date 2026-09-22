"""Entity marker utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from relationscope.constants import HEAD_END, HEAD_START, TAIL_END, TAIL_START
from relationscope.data.schema import EntitySpan


def _flatten_indices(span: object) -> list[int]:
    """Normalize FewRel's nested span representation to sorted token indices."""
    if not isinstance(span, list) or not span:
        raise ValueError(f"invalid FewRel span: {span!r}")

    candidate = span[0] if isinstance(span[0], list) else span
    if not isinstance(candidate, list) or not candidate:
        raise ValueError(f"invalid FewRel span: {span!r}")

    indices = sorted({int(index) for index in candidate})
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"non-contiguous entity span is not supported: {indices}")
    return indices


def mark_fewrel_tokens(
    tokens: Sequence[str],
    head_span: object,
    tail_span: object,
) -> list[str]:
    """Insert explicit entity markers into FewRel token sequences."""
    head = _flatten_indices(head_span)
    tail = _flatten_indices(tail_span)

    if set(head).intersection(tail):
        raise ValueError("head and tail spans overlap")
    if head[-1] >= len(tokens) or tail[-1] >= len(tokens):
        raise ValueError("entity span is outside token sequence")

    head_set = set(head)
    tail_set = set(tail)
    output: list[str] = []

    for index, token in enumerate(tokens):
        if index == head[0]:
            output.append(HEAD_START)
        if index == tail[0]:
            output.append(TAIL_START)

        output.append(str(token))

        if index == head[-1]:
            output.append(HEAD_END)
        if index == tail[-1]:
            output.append(TAIL_END)

    return output


def validate_character_span(text: str, span: EntitySpan) -> None:
    """Ensure offsets and supplied entity text agree with the source text."""
    if span.end > len(text):
        raise ValueError("entity span extends beyond source text")
    actual = text[span.start : span.end]
    if actual != span.text:
        raise ValueError(
            f"entity offsets do not match text: expected {span.text!r}, found {actual!r}"
        )


def mark_character_spans(text: str, head: EntitySpan, tail: EntitySpan) -> str:
    """Insert entity markers into raw text using validated character offsets."""
    validate_character_span(text, head)
    validate_character_span(text, tail)

    if max(head.start, tail.start) < min(head.end, tail.end):
        raise ValueError("head and tail spans overlap")

    insertions: list[tuple[int, str]] = [
        (head.start, f"{HEAD_START} "),
        (head.end, f" {HEAD_END}"),
        (tail.start, f"{TAIL_START} "),
        (tail.end, f" {TAIL_END}"),
    ]

    marked = text
    for position, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        marked = marked[:position] + marker + marked[position:]
    return " ".join(marked.split())


def first_unique_span(text: str, mention: str) -> EntitySpan:
    """Resolve one unique mention in text; reject missing or ambiguous mentions."""
    starts: list[int] = []
    cursor = 0
    while True:
        found = text.find(mention, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1

    if not starts:
        raise ValueError(f"mention not found: {mention!r}")
    if len(starts) > 1:
        raise ValueError(f"mention is ambiguous: {mention!r}")

    start = starts[0]
    return EntitySpan(text=mention, start=start, end=start + len(mention))
