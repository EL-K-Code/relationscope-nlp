"""Entity-aware Transformer relation classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from relationscope.constants import ENTITY_MARKERS, HEAD_START, TAIL_START


class EntityAwareRelationClassifier(nn.Module):
    """Pool CLS, HEAD, TAIL and |HEAD-TAIL| representations before classification."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        *,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 4, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_pos: torch.Tensor,
        tail_pos: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)

        cls = hidden[:, 0]
        head = hidden[batch, head_pos]
        tail = hidden[batch, tail_pos]
        features = torch.cat([cls, head, tail, torch.abs(head - tail)], dim=-1)
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
                    "architecture": "entity-aware-transformer",
                    "base_model": base_model,
                    "num_labels": len(label_map),
                    "markers": ENTITY_MARKERS,
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
        "EntityAwareRelationClassifier",
        PreTrainedTokenizerBase,
        dict[str, dict[str, str]],
    ]:
        label_map = json.loads(
            (bundle_dir / "relations.json").read_text(encoding="utf-8")
        )
        model = cls(str(bundle_dir / "encoder"), num_labels=len(label_map))
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
