"""Reproduce and correct the notebook's T5 relation-generation experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from relationscope.data.fewrel import load_closed_set_80
from relationscope.legacy.notebook_protocol import (
    NOTEBOOK_RELATIONS,
    NOTEBOOK_SEED,
    notebook_label,
    notebook_split,
)

RAW = Path("data/raw/fewrel")
OUT = Path("artifacts/legacy/t5")


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


class T5RelationDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        *,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        input_text = f"extract relation: {record.marked_text}"
        target_text = notebook_label(record)

        encoded = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument(
        "--legacy-target-length",
        action="store_true",
        help="Force the notebook's max_target_length=32 even if labels truncate.",
    )
    return parser.parse_args()


def _target_budget(tokenizer, records) -> tuple[int, int]:
    lengths = [
        len(tokenizer(notebook_label(record), add_special_tokens=True)["input_ids"])
        for record in records
    ]
    return max(lengths), sum(length > 32 for length in lengths)


def _truncate_gold(tokenizer, text: str, max_length: int) -> str:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)


@torch.inference_mode()
def evaluate(model, tokenizer, records, device, max_input_length, max_target_length):
    model.eval()
    full_gold_correct = 0
    truncated_gold_correct = 0
    predictions = []

    for record in tqdm(records, desc="t5-eval"):
        input_text = f"extract relation: {record.marked_text}"
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        generated = model.generate(
            **encoded,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
        )
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        full_gold = notebook_label(record)
        truncated_gold = _truncate_gold(tokenizer, full_gold, max_target_length)

        full_match = normalize(prediction) == normalize(full_gold)
        truncated_match = normalize(prediction) == normalize(truncated_gold)

        full_gold_correct += int(full_match)
        truncated_gold_correct += int(truncated_match)

        predictions.append(
            {
                "prediction": prediction,
                "gold_full": full_gold,
                "gold_truncated_to_generation_budget": truncated_gold,
                "full_gold_correct": full_match,
                "truncated_gold_correct": truncated_match,
            }
        )

    total = max(len(records), 1)
    return {
        "full_gold_exact_match": full_gold_correct / total,
        "truncated_gold_exact_match": truncated_gold_correct / total,
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()

    random.seed(NOTEBOOK_SEED)
    np.random.seed(NOTEBOOK_SEED)
    torch.manual_seed(NOTEBOOK_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(NOTEBOOK_SEED)

    records = load_closed_set_80(RAW)
    splits = notebook_split(records, NOTEBOOK_RELATIONS, seed=NOTEBOOK_SEED)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    required_target_length, labels_over_32 = _target_budget(tokenizer, splits["train"])

    max_target_length = 32 if args.legacy_target_length else required_target_length

    print(
        json.dumps(
            {
                "required_target_length": required_target_length,
                "labels_over_legacy_32": labels_over_32,
                "chosen_target_length": max_target_length,
            },
            indent=2,
        )
    )

    train_dataset = T5RelationDataset(
        splits["train"],
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=max_target_length,
    )
    validation_dataset = T5RelationDataset(
        splits["validation"],
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_dir = OUT / "best"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"t5 epoch {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(outputs.loss.item())

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                validation_loss += float(model(**batch).loss.item())

        train_loss /= max(len(train_loader), 1)
        validation_loss /= max(len(validation_loader), 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    best_model = T5ForConditionalGeneration.from_pretrained(best_dir).to(device)
    evaluation = evaluate(
        best_model,
        tokenizer,
        splits["test"],
        device,
        args.max_input_length,
        max_target_length,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "notebook_t5_generation",
        "model": args.model_name,
        "legacy_target_length_32": args.legacy_target_length,
        "required_target_length": required_target_length,
        "train_labels_over_32": labels_over_32,
        "max_target_length": max_target_length,
        "test_full_gold_exact_match": evaluation["full_gold_exact_match"],
        "test_truncated_gold_exact_match": evaluation["truncated_gold_exact_match"],
        "history": history,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (OUT / "test_predictions.json").write_text(
        json.dumps(evaluation["predictions"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
