"""Automatic entity discovery for RelationScope v3."""

from relationscope.document.analyzer import DocumentAnalyzer
from relationscope.document.entity_extractor import (
    EntityExtractor,
    EntityMentionCandidate,
    build_entity_extractor,
)

__all__ = [
    "DocumentAnalyzer",
    "EntityExtractor",
    "EntityMentionCandidate",
    "build_entity_extractor",
]
