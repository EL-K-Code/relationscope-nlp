"""Train the entity-aware Transformer classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from relationscope.data.io import read_jsonl
from relationscope.data.schema import RelationRecord
from relationscope.models.entity_classifier import (
    EntityAwareRelationClassifier,
    build_tokenizer,
    marker_positions,
    resize_for_markers,
)
from relationscope.training.metrics import classification_metrics

DATA = Path("data/processed/fewrel_closed_set_80")


class RelationDataset(Dataset[RelationRecord]):
    def __init__(self, records: list[RelationRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RelationRecord:
        return self.records[index]


class BatchCollator:
    def __init__(self, tokenizer: Any, rel2id: dict[str, int], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.max_length = max_length

    def __call__(self, records: list[RelationRecord]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [record.marked_text for record in records],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        head_pos, tail_pos = marker_positions(encoded["input_ids"], self.tokenizer)
        labels = torch.tensor(
            [self.rel2id[record.relation_id] for record in records],
            dtype=torch.long,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "head_pos": head_pos,
            "tail_pos": tail_pos,
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
    model: EntityAwareRelationClassifier,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        y = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs)
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())

    probs = np.concatenate(probabilities)
    gold = np.concatenate(labels)
    return classification_metrics(probs, gold), probs, gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train = read_jsonl(DATA / "train.jsonl")
    validation = read_jsonl(DATA / "validation.jsonl")
    test = read_jsonl(DATA / "test.jsonl")

    relation_ids = sorted({record.relation_id for record in train})
    rel2id = {relation_id: index for index, relation_id in enumerate(relation_ids)}
    metadata = {
        str(index): {
            "relation_id": relation_id,
            "relation_name": next(
                record.relation_name for record in train if record.relation_id == relation_id
            ),
        }
        for relation_id, index in rel2id.items()
    }

    tokenizer = build_tokenizer(args.model_name)
    model = EntityAwareRelationClassifier(args.model_name, num_labels=len(rel2id))
    resize_for_markers(model, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = BatchCollator(tokenizer, rel2id, args.max_length)
    train_loader = DataLoader(
        RelationDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        RelationDataset(validation),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_dir = args.output_dir / "best"
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        for batch in progress:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_metrics, _, _ = evaluate(model, validation_loader, device)
        epoch_report = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation": validation_metrics,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, indent=2))

        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            model.save_bundle(
                best_dir,
                tokenizer,
                metadata,
                base_model=args.model_name,
            )

    best_model, best_tokenizer, _ = EntityAwareRelationClassifier.load_bundle(
        best_dir,
        device=device,
    )
    best_collator = BatchCollator(best_tokenizer, rel2id, args.max_length)
    test_loader = DataLoader(
        RelationDataset(test),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=best_collator,
    )
    test_metrics, _, _ = evaluate(best_model, test_loader, device)

    report = {
        "protocol": "fewrel_closed_set_80",
        "base_model": args.model_name,
        "seed": args.seed,
        "best_validation_macro_f1": best_f1,
        "test": test_metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
