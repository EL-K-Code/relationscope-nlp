"""Automatic entity discovery for RelationScope v3."""

from relationscope.document.entity_extractor import (
    EntityExtractor,
    EntityMentionCandidate,
    build_entity_extractor,
)
from relationscope.document.analyzer import DocumentAnalyzer

__all__ = [
    "DocumentAnalyzer",
    "EntityExtractor",
    "EntityMentionCandidate",
    "build_entity_extractor",
]
