"""Production predictor for a saved RelationScope model bundle."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from relationscope.data.markers import mark_character_spans
from relationscope.data.schema import (
    BatchPredictExample,
    BatchPredictResponse,
    EntitySpan,
    ModelInfoResponse,
    Prediction,
    PredictionItem,
    PredictResponse,
)
from relationscope.models.entity_classifier import (
    EntityAwareRelationClassifier,
    marker_positions,
)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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
        self.model_id = bundle_dir.name

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_id=self.model_id,
            device=str(self.device),
            num_relations=len(self.label_map),
            max_length=self.max_length,
        )

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
        marked = mark_character_spans(text, head, tail)

        encoded = self.tokenizer(
            marked,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        head_pos, tail_pos = marker_positions(
            encoded["input_ids"],
            self.tokenizer,
        )

        logits = self.model(
            encoded["input_ids"],
            encoded["attention_mask"],
            head_pos,
            tail_pos,
        )
        predictions = self._decode_row(torch.softmax(logits[0], dim=-1), top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return PredictResponse(
            head=head.text,
            tail=tail.text,
            marked_text=marked,
            predictions=predictions,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        *,
        examples: list[BatchPredictExample],
        top_k: int,
    ) -> BatchPredictResponse:
        start_time = time.perf_counter()

        marked_texts = [
            mark_character_spans(example.text, example.head, example.tail)
            for example in examples
        ]
        encoded = self.tokenizer(
            marked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        head_pos, tail_pos = marker_positions(
            encoded["input_ids"],
            self.tokenizer,
        )

        logits = self.model(
            encoded["input_ids"],
            encoded["attention_mask"],
            head_pos,
            tail_pos,
        )
        probabilities = torch.softmax(logits, dim=-1)

        items = [
            PredictionItem(
                head=example.head.text,
                tail=example.tail.text,
                marked_text=marked,
                predictions=self._decode_row(row, top_k=top_k),
            )
            for example, marked, row in zip(
                examples,
                marked_texts,
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
