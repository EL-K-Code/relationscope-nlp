"""Pluggable automatic entity extraction used by RelationScope v3."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class EntityMentionCandidate:
    text: str
    start: int
    end: int
    label: str


class EntityExtractor(Protocol):
    @property
    def model_id(self) -> str: ...

    def extract(self, text: str) -> list[EntityMentionCandidate]: ...


class HeuristicEntityExtractor:
    """Deterministic zero-download fallback for local/offline development.

    It detects contiguous title-cased names, acronyms, and mixed organization-like
    spans. The transformer extractor should be preferred for production document
    intelligence when a reviewed NER bundle is available.
    """

    _pattern = re.compile(
        r"""
        (?<![\w])
        (
            (?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]{1,})
            (?:
                \s+
                (?:
                    [A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]{1,}
                    |(?:de|du|des|la|le|of|the|and|&|University|Institute|Inc|Ltd|Corp)
                )
            ){0,5}
            |
            (?:[A-Z]{2,}(?:[-&][A-Z]{2,})*)
        )
        """,
        re.VERBOSE,
    )

    _sentence_boundary = re.compile(r"(?<=[.!?])(?:\s+|$)")
    _trailing_sentence_punctuation = ".,;:!?"

    @property
    def model_id(self) -> str:
        return "heuristic-capitalized-entities-v1"

    def extract(self, text: str) -> list[EntityMentionCandidate]:
        candidates: list[EntityMentionCandidate] = []
        seen: set[tuple[int, int]] = set()

        segment_start = 0
        segment_ranges: list[tuple[int, int]] = []
        for boundary in self._sentence_boundary.finditer(text):
            segment_ranges.append((segment_start, boundary.start()))
            segment_start = boundary.end()
        if segment_start < len(text):
            segment_ranges.append((segment_start, len(text)))
        elif not segment_ranges and text:
            segment_ranges.append((0, len(text)))

        for segment_start, segment_end in segment_ranges:
            segment = text[segment_start:segment_end]
            for match in self._pattern.finditer(segment):
                raw = match.group(1).strip()
                raw = raw.rstrip(self._trailing_sentence_punctuation)
                if len(raw) < 2:
                    continue

                start = segment_start + match.start(1)
                end = start + len(raw)
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)

                label = "ORG" if any(
                    marker in raw
                    for marker in (
                        "University",
                        "Institute",
                        "Inc",
                        "Ltd",
                        "Corp",
                        "Company",
                        "Bank",
                    )
                ) else "ENTITY"
                candidates.append(
                    EntityMentionCandidate(
                        text=raw,
                        start=start,
                        end=end,
                        label=label,
                    )
                )
        return candidates


class TransformerEntityExtractor:
    """Hugging Face token-classification NER with aggregated entity spans."""

    def __init__(self, model_name_or_path: str, *, device: str = "auto") -> None:
        from transformers import pipeline

        if device == "auto":
            pipeline_device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            pipeline_device = -1
        elif device.startswith("cuda"):
            pipeline_device = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
        else:
            pipeline_device = -1

        self._model_id = model_name_or_path
        self._pipeline = pipeline(
            "token-classification",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            aggregation_strategy="simple",
            device=pipeline_device,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, text: str) -> list[EntityMentionCandidate]:
        raw_results = self._pipeline(text)
        candidates: list[EntityMentionCandidate] = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            start = raw.get("start")
            end = raw.get("end")
            label = raw.get("entity_group") or raw.get("entity") or "ENTITY"
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue
            surface = text[start:end]
            if not surface.strip():
                continue
            candidates.append(
                EntityMentionCandidate(
                    text=surface,
                    start=start,
                    end=end,
                    label=str(label),
                )
            )
        return candidates


def build_entity_extractor() -> EntityExtractor:
    """Build the configured extractor.

    RELATIONSCOPE_NER_MODEL:
      - empty / unset: deterministic heuristic fallback
      - any value: Hugging Face/local token-classification model path or ID
    """
    model_name = os.getenv("RELATIONSCOPE_NER_MODEL", "").strip()
    if not model_name:
        return HeuristicEntityExtractor()
    return TransformerEntityExtractor(
        model_name,
        device=os.getenv("RELATIONSCOPE_NER_DEVICE", "auto"),
    )
