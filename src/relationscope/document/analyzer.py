"""Document-level entity discovery, relation extraction and graph assembly."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass

from relationscope.data.schema import BatchPredictExample, EntitySpan
from relationscope.data.v3_schema import (
    DocumentAnalyzeRequest,
    DocumentAnalyzeResponse,
    DocumentEntity,
    DocumentRelation,
    EntityMention,
)
from relationscope.document.entity_extractor import EntityExtractor, EntityMentionCandidate
from relationscope.inference.predictor import RelationPredictor
from relationscope.ops.active_learning import ActiveLearningService


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    index: int
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[SentenceSpan]:
    """Return stable sentence spans while preserving document offsets."""
    spans: list[SentenceSpan] = []
    cursor = 0
    pattern = re.compile(r"(?<=[.!?])(?:\s+|$)")

    for match in pattern.finditer(text):
        end = match.start()
        segment = text[cursor:end].strip()
        if segment:
            left_trim = len(text[cursor:end]) - len(text[cursor:end].lstrip())
            start = cursor + left_trim
            spans.append(
                SentenceSpan(
                    index=len(spans),
                    text=text[start:end].rstrip(),
                    start=start,
                    end=end,
                )
            )
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        left_trim = len(text[cursor:]) - len(text[cursor:].lstrip())
        start = cursor + left_trim
        spans.append(
            SentenceSpan(
                index=len(spans),
                text=text[start : len(text)].rstrip(),
                start=start,
                end=len(text),
            )
        )

    if not spans and text.strip():
        stripped = text.strip()
        start = text.index(stripped)
        spans.append(
            SentenceSpan(
                index=0,
                text=stripped,
                start=start,
                end=start + len(stripped),
            )
        )
    return spans


def _normalize_entity(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    return normalized.strip(".,;:!?()[]{}\"'")


def _entity_id(normalized_text: str, label: str) -> str:
    digest = hashlib.sha1(
        f"{label.casefold()}::{normalized_text}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"ent_{digest}"


def _edge_id(
    head_entity_id: str,
    relation_id: str,
    tail_entity_id: str,
    sentence_index: int,
) -> str:
    digest = hashlib.sha1(
        (
            f"{head_entity_id}::{relation_id}::{tail_entity_id}::"
            f"{sentence_index}"
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:18]
    return f"edge_{digest}"


class DocumentAnalyzer:
    """Turn raw text into entities, relation edges and review candidates."""

    def __init__(
        self,
        *,
        predictor: RelationPredictor,
        entity_extractor: EntityExtractor,
        active_learning: ActiveLearningService | None = None,
    ) -> None:
        self.predictor = predictor
        self.entity_extractor = entity_extractor
        self.active_learning = active_learning

    def _sentence_for_mention(
        self,
        mention: EntityMentionCandidate,
        sentences: list[SentenceSpan],
    ) -> SentenceSpan | None:
        for sentence in sentences:
            if mention.start >= sentence.start and mention.end <= sentence.end:
                return sentence
        return None

    def analyze(self, request: DocumentAnalyzeRequest) -> DocumentAnalyzeResponse:
        started = time.perf_counter()
        document_id = f"doc_{uuid.uuid4().hex}"
        sentences = split_sentences(request.text)
        mentions = self.entity_extractor.extract(request.text)

        canonical_mentions: dict[
            tuple[str, str],
            list[tuple[EntityMentionCandidate, SentenceSpan]],
        ] = defaultdict(list)
        sentence_mentions: dict[
            int,
            list[tuple[EntityMentionCandidate, str]],
        ] = defaultdict(list)

        for mention in mentions:
            sentence = self._sentence_for_mention(mention, sentences)
            normalized = _normalize_entity(mention.text)
            if sentence is None or not normalized:
                continue
            key = (normalized, mention.label)
            entity_id = _entity_id(normalized, mention.label)
            canonical_mentions[key].append((mention, sentence))
            sentence_mentions[sentence.index].append((mention, entity_id))

        entities: list[DocumentEntity] = []
        for (normalized, label), grouped in canonical_mentions.items():
            representative = grouped[0][0].text
            entity_id = _entity_id(normalized, label)
            entities.append(
                DocumentEntity(
                    entity_id=entity_id,
                    text=representative,
                    normalized_text=normalized,
                    label=label,
                    mentions=[
                        EntityMention(
                            sentence_index=sentence.index,
                            start=mention.start,
                            end=mention.end,
                        )
                        for mention, sentence in grouped
                    ],
                )
            )
        entities.sort(key=lambda item: item.mentions[0].start)

        candidates: list[
            tuple[
                BatchPredictExample,
                SentenceSpan,
                EntityMentionCandidate,
                EntityMentionCandidate,
                str,
                str,
            ]
        ] = []

        for sentence in sentences:
            local_mentions = sentence_mentions.get(sentence.index, [])
            for head_index, (head, head_id) in enumerate(local_mentions):
                for tail_index, (tail, tail_id) in enumerate(local_mentions):
                    if head_index == tail_index or head_id == tail_id:
                        continue
                    if len(candidates) >= request.max_candidate_pairs:
                        break

                    head_span = EntitySpan(
                        text=head.text,
                        start=head.start - sentence.start,
                        end=head.end - sentence.start,
                    )
                    tail_span = EntitySpan(
                        text=tail.text,
                        start=tail.start - sentence.start,
                        end=tail.end - sentence.start,
                    )
                    candidates.append(
                        (
                            BatchPredictExample(
                                text=sentence.text,
                                head=head_span,
                                tail=tail_span,
                            ),
                            sentence,
                            head,
                            tail,
                            head_id,
                            tail_id,
                        )
                    )
                if len(candidates) >= request.max_candidate_pairs:
                    break
            if len(candidates) >= request.max_candidate_pairs:
                break

        responses = self.predictor.predict_open_set_batch(
            examples=[item[0] for item in candidates],
            top_k=1,
            profile=request.profile,
        )

        relations: list[DocumentRelation] = []
        review_candidates = 0
        queued_for_review = 0

        for candidate, response in zip(candidates, responses, strict=True):
            example, sentence, head, tail, head_id, tail_id = candidate
            if not response.predictions:
                continue
            top = response.predictions[0]

            if self.active_learning is not None:
                if self.active_learning.should_queue(response):
                    review_candidates += 1
                    queued = self.active_learning.enqueue(
                        text=sentence.text,
                        head=example.head,
                        tail=example.tail,
                        response=response,
                        source_document_id=document_id,
                        consent_to_store_review_data=(
                            request.consent_to_store_review_data
                        ),
                    )
                    if queued is not None:
                        queued_for_review += 1

            if not response.accepted and not request.include_unknown:
                continue

            relations.append(
                DocumentRelation(
                    edge_id=_edge_id(
                        head_id,
                        top.relation_id,
                        tail_id,
                        sentence.index,
                    ),
                    head_entity_id=head_id,
                    tail_entity_id=tail_id,
                    head_text=head.text,
                    tail_text=tail.text,
                    relation=top,
                    decision=response.decision,
                    accepted=response.accepted,
                    profile=response.profile,
                    threshold=response.threshold,
                    sentence_index=sentence.index,
                    evidence_text=sentence.text,
                )
            )

        # Keep only the strongest duplicate canonical edge for the same sentence.
        best_by_edge: dict[str, DocumentRelation] = {}
        for relation in relations:
            current = best_by_edge.get(relation.edge_id)
            if (
                current is None
                or relation.relation.confidence > current.relation.confidence
            ):
                best_by_edge[relation.edge_id] = relation

        final_relations = sorted(
            best_by_edge.values(),
            key=lambda item: (
                item.sentence_index,
                -item.relation.confidence,
                item.head_text,
                item.tail_text,
            ),
        )

        return DocumentAnalyzeResponse(
            document_id=document_id,
            title=request.title,
            model_id=self.predictor.model_id,
            ner_model_id=self.entity_extractor.model_id,
            entities=entities,
            relations=final_relations,
            candidate_pairs=len(candidates),
            processed_pairs=len(responses),
            review_candidates=review_candidates,
            queued_for_review=queued_for_review,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
