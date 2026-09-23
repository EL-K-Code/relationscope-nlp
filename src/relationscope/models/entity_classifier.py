"""Transformer relation classifiers used by RelationScope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from relationscope.constants import ENTITY_MARKERS, HEAD_START, TAIL_START

PoolingMode = Literal["cls", "entity_pair"]
InputView = Literal["raw", "marked"]


class EntityAwareRelationClassifier(nn.Module):
    """Transformer classifier with configurable entity-aware pooling."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        *,
        dropout: float = 0.2,
        pooling_mode: PoolingMode = "entity_pair",
        input_view: InputView = "marked",
    ) -> None:
        super().__init__()
        if pooling_mode not in {"cls", "entity_pair"}:
            raise ValueError(f"unsupported pooling mode: {pooling_mode}")
        if input_view not in {"raw", "marked"}:
            raise ValueError(f"unsupported input view: {input_view}")
        if pooling_mode == "entity_pair" and input_view != "marked":
            raise ValueError("entity_pair pooling requires marked input")

        self.pooling_mode = pooling_mode
        self.input_view = input_view
        self.dropout_probability = float(dropout)

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden = int(self.encoder.config.hidden_size)
        feature_size = hidden if pooling_mode == "cls" else hidden * 4

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_pos: torch.Tensor | None = None,
        tail_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]

        if self.pooling_mode == "cls":
            features = cls
        else:
            if head_pos is None or tail_pos is None:
                raise ValueError("entity_pair pooling requires HEAD and TAIL positions")
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            head = hidden[batch, head_pos]
            tail = hidden[batch, tail_pos]
            features = torch.cat(
                [cls, head, tail, torch.abs(head - tail)],
                dim=-1,
            )

        return self.classifier(self.dropout(features))

    def save_bundle(
        self,
        output_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
        label_map: dict[str, dict[str, str]],
        *,
        base_model: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        torch.save(self.classifier.state_dict(), output_dir / "classifier.pt")

        (output_dir / "relations.json").write_text(
            json.dumps(label_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "architecture": "transformer-relation-classifier",
                    "base_model": base_model,
                    "num_labels": len(label_map),
                    "markers": ENTITY_MARKERS,
                    "pooling_mode": self.pooling_mode,
                    "input_view": self.input_view,
                    "dropout": self.dropout_probability,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_bundle(
        cls,
        bundle_dir: Path,
        *,
        device: torch.device,
    ) -> tuple[
        EntityAwareRelationClassifier,
        PreTrainedTokenizerBase,
        dict[str, dict[str, str]],
    ]:
        label_map = json.loads(
            (bundle_dir / "relations.json").read_text(encoding="utf-8")
        )

        config_path = bundle_dir / "model_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        pooling_mode = config.get("pooling_mode", "entity_pair")
        input_view = config.get("input_view", "marked")
        dropout = float(config.get("dropout", 0.2))

        model = cls(
            str(bundle_dir / "encoder"),
            num_labels=len(label_map),
            pooling_mode=pooling_mode,
            input_view=input_view,
            dropout=dropout,
        )
        classifier_state = torch.load(
            bundle_dir / "classifier.pt",
            map_location=device,
            weights_only=True,
        )
        model.classifier.load_state_dict(classifier_state)

        tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "tokenizer")
        model.to(device)
        model.eval()
        return model, tokenizer, label_map


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})
    return tokenizer


def resize_for_markers(
    model: EntityAwareRelationClassifier,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    model.encoder.resize_token_embeddings(len(tokenizer))


def marker_positions(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find exactly one HEAD and one TAIL start marker in every encoded example."""
    head_id = tokenizer.convert_tokens_to_ids(HEAD_START)
    tail_id = tokenizer.convert_tokens_to_ids(TAIL_START)

    heads: list[int] = []
    tails: list[int] = []
    for row in input_ids:
        head_matches = (row == head_id).nonzero(as_tuple=False).flatten()
        tail_matches = (row == tail_id).nonzero(as_tuple=False).flatten()
        if len(head_matches) != 1 or len(tail_matches) != 1:
            raise ValueError(
                "each encoded example must contain exactly one HEAD and TAIL marker"
            )
        heads.append(int(head_matches.item()))
        tails.append(int(tail_matches.item()))

    return (
        torch.tensor(heads, dtype=torch.long, device=input_ids.device),
        torch.tensor(tails, dtype=torch.long, device=input_ids.device),
    )
