"""Production predictor for a saved RelationScope model bundle."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from relationscope.constants import ENTITY_MARKERS
from relationscope.data.markers import mark_character_spans
from relationscope.data.schema import (
    BatchPredictExample,
    BatchPredictResponse,
    EntitySpan,
    ExplainResponse,
    ExplanationToken,
    ModelInfoResponse,
    OpenSetPredictResponse,
    OpenSetProfile,
    Prediction,
    PredictionItem,
    PredictResponse,
)
from relationscope.data.v2_schema import (
    InfluenceExplainResponse,
    RelationCatalogItem,
    RelationCatalogResponse,
    WordInfluence,
)
from relationscope.models.entity_classifier import (
    EntityAwareRelationClassifier,
    marker_positions,
)


def resolve_model_id(bundle_dir: Path) -> str:
    """Return the promoted model version from manifest.json when available."""
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        model_version = manifest.get("model_version")
        if isinstance(model_version, str) and model_version.strip():
            return model_version.strip()
    return bundle_dir.name


SUPPORTED_OPEN_SET_PROFILES: tuple[OpenSetProfile, ...] = (
    "high_coverage",
    "balanced",
    "high_safety",
)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def available_open_set_profiles(
    config: dict[str, object] | None,
) -> list[OpenSetProfile]:
    """Return configured operating profiles, preserving the public profile order."""
    if config is None:
        return []

    configured = config.get("profiles")
    available: list[OpenSetProfile] = []

    if isinstance(configured, dict):
        for profile in SUPPORTED_OPEN_SET_PROFILES:
            entry = configured.get(profile)
            if isinstance(entry, dict) and "threshold" in entry:
                available.append(profile)

    if not available and "abstention_threshold" in config:
        available.append("balanced")

    return available


def default_open_set_profile(
    config: dict[str, object] | None,
) -> OpenSetProfile | None:
    """Resolve the default profile while remaining compatible with legacy bundles."""
    available = available_open_set_profiles(config)
    if not available or config is None:
        return None

    requested = config.get("default_profile")
    for profile in available:
        if requested == profile:
            return profile

    if "balanced" in available:
        return "balanced"
    return available[0]


def resolve_open_set_threshold(
    config: dict[str, object] | None,
    profile: OpenSetProfile,
) -> float:
    """Resolve and validate the threshold for one operating profile."""
    if config is None:
        raise RuntimeError("open-set calibration is not available for this model")

    configured = config.get("profiles")
    raw_threshold: object | None = None

    if isinstance(configured, dict):
        entry = configured.get(profile)
        if isinstance(entry, dict):
            raw_threshold = entry.get("threshold")

    if raw_threshold is None and profile == "balanced":
        raw_threshold = config.get("abstention_threshold")

    if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
        raise RuntimeError(
            f"open-set profile {profile!r} is not configured in this model bundle"
        )

    threshold = float(raw_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"invalid abstention threshold for profile {profile!r}: {threshold}"
        )
    return threshold


class RelationPredictor:
    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str = "auto",
        max_length: int = 160,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.device = resolve_device(device)
        self.max_length = max_length
        loaded = EntityAwareRelationClassifier.load_bundle(
            bundle_dir,
            device=self.device,
        )
        self.model, self.tokenizer, self.label_map = loaded
        self.model_id = resolve_model_id(bundle_dir)

        open_set_path = bundle_dir / "open_set_config.json"
        self.open_set_config: dict[str, object] | None = (
            json.loads(open_set_path.read_text(encoding="utf-8"))
            if open_set_path.exists()
            else None
        )

    def relation_catalog(self) -> RelationCatalogResponse:
        items = [
            RelationCatalogItem(
                relation_id=metadata["relation_id"],
                relation_name=metadata["relation_name"],
            )
            for _, metadata in sorted(
                self.label_map.items(),
                key=lambda item: int(item[0]),
            )
        ]
        return RelationCatalogResponse(items=items)

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_id=self.model_id,
            device=str(self.device),
            num_relations=len(self.label_map),
            max_length=self.max_length,
            pooling_mode=self.model.pooling_mode,
            input_view=self.model.input_view,
            open_set_enabled=self.open_set_config is not None,
            default_open_set_profile=default_open_set_profile(self.open_set_config),
            open_set_profiles=available_open_set_profiles(self.open_set_config),
        )

    def _input_text(self, text: str, head: EntitySpan, tail: EntitySpan) -> str:
        if self.model.input_view == "marked":
            return mark_character_spans(text, head, tail)
        return text

    def _model_inputs(
        self,
        encoded: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if self.model.pooling_mode == "entity_pair":
            head_pos, tail_pos = marker_positions(
                encoded["input_ids"],
                self.tokenizer,
            )
            inputs["head_pos"] = head_pos
            inputs["tail_pos"] = tail_pos
        return inputs

    def _decode_row(
        self,
        probabilities: torch.Tensor,
        *,
        top_k: int,
    ) -> list[Prediction]:
        count = min(top_k, int(probabilities.shape[0]))
        values, indices = torch.topk(probabilities, k=count)

        predictions: list[Prediction] = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            metadata = self.label_map[str(index)]
            predictions.append(
                Prediction(
                    relation_id=metadata["relation_id"],
                    relation_name=metadata["relation_name"],
                    confidence=float(value),
                )
            )
        return predictions

    def _probabilities(
        self,
        logits: torch.Tensor,
        *,
        calibrated: bool = False,
    ) -> torch.Tensor:
        if calibrated:
            if self.open_set_config is None:
                raise RuntimeError("open-set calibration is not available for this model")
            temperature = float(self.open_set_config["temperature"])
            logits = logits / temperature
        return torch.softmax(logits, dim=-1)

    def _relation_index(self, relation_id: str) -> int:
        for raw_index, metadata in self.label_map.items():
            if metadata["relation_id"] == relation_id:
                return int(raw_index)
        raise ValueError(f"unknown relation_id: {relation_id}")

    @torch.inference_mode()
    def predict(
        self,
        *,
        text: str,
        head: EntitySpan,
        tail: EntitySpan,
        top_k: int,
    ) -> PredictResponse:
        start_time = time.perf_counter()
        model_text = self._input_text(text, head, tail)

        encoded = self.tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**self._model_inputs(encoded))
        predictions = self._decode_row(
            self._probabilities(logits[0]),
            top_k=top_k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        return PredictResponse(
            head=head.text,
            tail=tail.text,
            marked_text=model_text,
            predictions=predictions,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )

    @torch.inference_mode()
    def predict_open_set(
        self,
        *,
        text: str,
        head: EntitySpan,
        tail: EntitySpan,
        top_k: int,
        profile: OpenSetProfile = "balanced",
    ) -> OpenSetPredictResponse:
        if self.open_set_config is None:
            raise RuntimeError("this model bundle has no open-set calibration")

        start_time = time.perf_counter()
        model_text = self._input_text(text, head, tail)
        encoded = self.tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**self._model_inputs(encoded))
        probabilities = self._probabilities(logits[0], calibrated=True)
        predictions = self._decode_row(probabilities, top_k=top_k)

        threshold = resolve_open_set_threshold(self.open_set_config, profile)
        temperature = float(self.open_set_config["temperature"])
        accepted = float(probabilities.max().item()) >= threshold

        return OpenSetPredictResponse(
            head=head.text,
            tail=tail.text,
            marked_text=model_text,
            predictions=predictions,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            model_id=self.model_id,
            decision="KNOWN_RELATION" if accepted else "UNKNOWN_RELATION",
            accepted=accepted,
            profile=profile,
            threshold=threshold,
            temperature=temperature,
        )

    @torch.inference_mode()
    def predict_open_set_batch(
        self,
        *,
        examples: list[BatchPredictExample],
        top_k: int,
        profile: OpenSetProfile = "balanced",
    ) -> list[OpenSetPredictResponse]:
        """Vectorized calibrated open-set inference for one bounded chunk."""
        if self.open_set_config is None:
            raise RuntimeError("this model bundle has no open-set calibration")
        if not examples:
            return []

        started = time.perf_counter()
        model_texts = [
            self._input_text(example.text, example.head, example.tail)
            for example in examples
        ]
        encoded = self.tokenizer(
            model_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**self._model_inputs(encoded))
        probabilities = self._probabilities(logits, calibrated=True)

        threshold = resolve_open_set_threshold(self.open_set_config, profile)
        temperature = float(self.open_set_config["temperature"])
        per_item_latency = ((time.perf_counter() - started) * 1000) / len(examples)

        responses: list[OpenSetPredictResponse] = []
        for example, model_text, row in zip(
            examples,
            model_texts,
            probabilities,
            strict=True,
        ):
            predictions = self._decode_row(row, top_k=top_k)
            accepted = float(row.max().item()) >= threshold
            responses.append(
                OpenSetPredictResponse(
                    head=example.head.text,
                    tail=example.tail.text,
                    marked_text=model_text,
                    predictions=predictions,
                    latency_ms=per_item_latency,
                    model_id=self.model_id,
                    decision="KNOWN_RELATION" if accepted else "UNKNOWN_RELATION",
                    accepted=accepted,
                    profile=profile,
                    threshold=threshold,
                    temperature=temperature,
                )
            )
        return responses

    @torch.inference_mode()
    def explain(
        self,
        *,
        text: str,
        head: EntitySpan,
        tail: EntitySpan,
        top_evidence: int,
    ) -> ExplainResponse:
        """Legacy v1 sub-token occlusion explanation."""
        start_time = time.perf_counter()
        model_text = self._input_text(text, head, tail)
        encoded = self.tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        baseline_logits = self.model(**self._model_inputs(encoded))
        baseline_probabilities = self._probabilities(baseline_logits[0])
        predicted_index = int(torch.argmax(baseline_probabilities).item())
        baseline_confidence = float(baseline_probabilities[predicted_index].item())

        token_ids = encoded["input_ids"][0]
        marker_ids = {
            self.tokenizer.convert_tokens_to_ids(marker)
            for marker in ENTITY_MARKERS
        }
        excluded = set(self.tokenizer.all_special_ids).union(marker_ids)

        candidate_positions = [
            position
            for position, token_id in enumerate(token_ids.tolist())
            if token_id not in excluded
        ]

        replacement_id = self.tokenizer.mask_token_id
        if replacement_id is None:
            replacement_id = self.tokenizer.unk_token_id
        if replacement_id is None:
            raise RuntimeError("tokenizer has neither mask_token nor unk_token")

        evidence: list[ExplanationToken] = []
        if candidate_positions:
            occluded_ids = token_ids.repeat(len(candidate_positions), 1)
            for row, position in enumerate(candidate_positions):
                occluded_ids[row, position] = replacement_id

            occluded_attention = encoded["attention_mask"].repeat(
                len(candidate_positions),
                1,
            )
            occluded = {
                "input_ids": occluded_ids,
                "attention_mask": occluded_attention,
            }
            occluded_logits = self.model(**self._model_inputs(occluded))
            occluded_probabilities = self._probabilities(occluded_logits)

            for row, position in enumerate(candidate_positions):
                confidence = float(
                    occluded_probabilities[row, predicted_index].item()
                )
                token = self.tokenizer.convert_ids_to_tokens(
                    int(token_ids[position].item())
                )
                evidence.append(
                    ExplanationToken(
                        token=str(token),
                        position=position,
                        confidence_drop=baseline_confidence - confidence,
                    )
                )

        evidence.sort(key=lambda item: item.confidence_drop, reverse=True)
        metadata = self.label_map[str(predicted_index)]

        return ExplainResponse(
            model_id=self.model_id,
            relation=Prediction(
                relation_id=metadata["relation_id"],
                relation_name=metadata["relation_name"],
                confidence=baseline_confidence,
            ),
            baseline_confidence=baseline_confidence,
            evidence=evidence[:top_evidence],
            method="token_occlusion",
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    @torch.inference_mode()
    def explain_words(
        self,
        *,
        text: str,
        head: EntitySpan,
        tail: EntitySpan,
        profile: OpenSetProfile = "balanced",
        target_relation_id: str | None = None,
    ) -> InfluenceExplainResponse:
        """Return a signed, relation-specific word influence map.

        Each displayed word is masked as a complete group of tokenizer pieces. The
        score is the drop in calibrated target-relation confidence caused by that
        intervention. Positive values support the relation; negative values oppose it.
        """
        if self.open_set_config is None:
            raise RuntimeError("this model bundle has no open-set calibration")
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("word influence requires a fast tokenizer")

        started = time.perf_counter()
        model_text = self._input_text(text, head, tail)
        raw_encoded = self.tokenizer(
            model_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = raw_encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in raw_encoded.items()}

        baseline_logits = self.model(**self._model_inputs(encoded))
        baseline_probabilities = self._probabilities(
            baseline_logits[0],
            calibrated=True,
        )
        predicted_index = int(torch.argmax(baseline_probabilities).item())
        target_index = (
            self._relation_index(target_relation_id)
            if target_relation_id is not None
            else predicted_index
        )
        baseline_confidence = float(baseline_probabilities[target_index].item())

        threshold = resolve_open_set_threshold(self.open_set_config, profile)
        accepted = float(baseline_probabilities.max().item()) >= threshold

        marker_ids = {
            self.tokenizer.convert_tokens_to_ids(marker)
            for marker in ENTITY_MARKERS
        }
        excluded = set(self.tokenizer.all_special_ids).union(marker_ids)
        token_ids = encoded["input_ids"][0]

        replacement_id = self.tokenizer.mask_token_id
        if replacement_id is None:
            replacement_id = self.tokenizer.unk_token_id
        if replacement_id is None:
            raise RuntimeError("tokenizer has neither mask_token nor unk_token")

        groups: list[tuple[str, int, int, list[int]]] = []
        for match in re.finditer(r"\S+", model_text):
            word = match.group(0)
            if word in ENTITY_MARKERS:
                continue

            positions: list[int] = []
            for position, ((start, end), token_id) in enumerate(
                zip(offsets, token_ids.tolist(), strict=True)
            ):
                if token_id in excluded or start == end == 0:
                    continue
                if start < match.end() and end > match.start():
                    positions.append(position)

            if positions:
                groups.append((word, match.start(), match.end(), positions))

        drops: list[float] = []
        if groups:
            occluded_ids = token_ids.repeat(len(groups), 1)
            for row, (_, _, _, positions) in enumerate(groups):
                for position in positions:
                    occluded_ids[row, position] = replacement_id

            occluded = {
                "input_ids": occluded_ids,
                "attention_mask": encoded["attention_mask"].repeat(len(groups), 1),
            }
            occluded_logits = self.model(**self._model_inputs(occluded))
            occluded_probabilities = self._probabilities(
                occluded_logits,
                calibrated=True,
            )
            drops = [
                baseline_confidence
                - float(occluded_probabilities[row, target_index].item())
                for row in range(len(groups))
            ]

        max_abs = max((abs(value) for value in drops), default=0.0)
        influences: list[WordInfluence] = []
        for (word, start, end, _), drop in zip(groups, drops, strict=True):
            normalized = drop / max_abs if max_abs > 0 else 0.0
            direction = (
                "supports"
                if drop > 1e-6
                else "opposes"
                if drop < -1e-6
                else "neutral"
            )
            influences.append(
                WordInfluence(
                    word=word,
                    start=start,
                    end=end,
                    confidence_drop=drop,
                    normalized_influence=max(-1.0, min(1.0, normalized)),
                    direction=direction,
                )
            )

        metadata = self.label_map[str(target_index)]
        return InfluenceExplainResponse(
            model_id=self.model_id,
            relation=Prediction(
                relation_id=metadata["relation_id"],
                relation_name=metadata["relation_name"],
                confidence=baseline_confidence,
            ),
            decision="KNOWN_RELATION" if accepted else "UNKNOWN_RELATION",
            profile=profile,
            threshold=threshold,
            baseline_confidence=baseline_confidence,
            marked_text=model_text,
            influences=influences,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        *,
        examples: list[BatchPredictExample],
        top_k: int,
    ) -> BatchPredictResponse:
        start_time = time.perf_counter()

        model_texts = [
            self._input_text(example.text, example.head, example.tail)
            for example in examples
        ]
        encoded = self.tokenizer(
            model_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**self._model_inputs(encoded))
        probabilities = self._probabilities(logits)

        items = [
            PredictionItem(
                head=example.head.text,
                tail=example.tail.text,
                marked_text=model_text,
                predictions=self._decode_row(row, top_k=top_k),
            )
            for example, model_text, row in zip(
                examples,
                model_texts,
                probabilities,
                strict=True,
            )
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return BatchPredictResponse(
            items=items,
            latency_ms=latency_ms,
            model_id=self.model_id,
            batch_size=len(items),
        )
