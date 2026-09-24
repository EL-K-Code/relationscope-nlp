"""Conservative cleanup for text extracted from uploaded PDFs."""

from __future__ import annotations

import re
import unicodedata

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
_DEHYPHENATE_LINE_BREAK = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_BLANK_LINES = re.compile(r"\n{3,}")
_SENTENCE_TERMINATORS = (".", "!", "?")


def _clean_line(line: str) -> str:
    return _HORIZONTAL_WHITESPACE.sub(" ", line).strip()


def _should_join_wrapped_line(previous: str, current: str) -> bool:
    """Return True only for likely visual wrapping, not headings/table labels."""
    if not previous or not current:
        return False
    if previous.endswith(_SENTENCE_TERMINATORS):
        return False
    if previous.endswith((",", ";", ":")):
        return True
    if len(previous) >= 72:
        return True
    return current[:1].islower()


def normalize_pdf_pages(pages: list[str]) -> str:
    """Normalize PDF text while preserving meaningful layout boundaries.

    The cleanup is intentionally conservative:
    - Unicode and invisible-character normalization;
    - de-hyphenation across visual line wraps;
    - horizontal whitespace collapse;
    - joining only likely wrapped prose lines;
    - preservation of short standalone lines so downstream logic can
      distinguish prose from headings/table labels.
    """
    normalized_pages: list[str] = []

    for page in pages:
        text = unicodedata.normalize("NFKC", page)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00ad", "").replace("\u200b", "")
        text = _DEHYPHENATE_LINE_BREAK.sub("", text)

        output: list[str] = []
        pending = ""

        for raw_line in text.split("\n"):
            line = _clean_line(raw_line)
            if not line:
                if pending:
                    output.append(pending)
                    pending = ""
                if output and output[-1] != "":
                    output.append("")
                continue

            if not pending:
                pending = line
                continue

            if _should_join_wrapped_line(pending, line):
                pending = f"{pending} {line}"
            else:
                output.append(pending)
                pending = line

        if pending:
            output.append(pending)

        page_text = "\n".join(output).strip()
        page_text = _BLANK_LINES.sub("\n\n", page_text)
        if page_text:
            normalized_pages.append(page_text)

    return "\n\n".join(normalized_pages).strip()
