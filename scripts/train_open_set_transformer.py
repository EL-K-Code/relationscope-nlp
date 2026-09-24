"""Train and evaluate RelationScope under a strict unseen-relation protocol."""

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
from relationscope.data.markers import mark_character_spans
from relationscope.data.schema import RelationRecord
from relationscope.data.v3_schema import TrainingCandidate
from relationscope.models.entity_classifier import (
    EntityAwareRelationClassifier,
    build_tokenizer,
    marker_positions,
    resize_for_markers,
)
from relationscope.training.calibration import (
    TemperatureScaler,
    softmax_with_temperature,
)
from relationscope.training.metrics import classification_metrics
from relationscope.training.open_set_metrics import (
    open_set_metrics,
    risk_coverage_curve,
    tune_abstention_threshold,
)

DATA = Path("data/processed/fewrel_open_set")


class RelationDataset(Dataset[RelationRecord]):
    def __init__(self, records: list[RelationRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RelationRecord:
        return self.records[index]


class KnownCollator:
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
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "head_pos": head_pos,
            "tail_pos": tail_pos,
            "labels": torch.tensor(
                [self.rel2id[record.relation_id] for record in records],
                dtype=torch.long,
            ),
        }


class ScoreCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
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
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "head_pos": head_pos,
            "tail_pos": tail_pos,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def logits_and_labels(
    model: EntityAwareRelationClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="score-known", leave=False):
        y = batch.pop("labels").numpy()
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits.append(model(**inputs).cpu().numpy())
        labels.append(y)

    return np.concatenate(logits), np.concatenate(labels)


@torch.inference_mode()
def logits_only(
    model: EntityAwareRelationClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits: list[np.ndarray] = []

    for batch in tqdm(loader, desc="score-unknown", leave=False):
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits.append(model(**inputs).cpu().numpy())

    return np.concatenate(logits)


def load_review_snapshot(
    path: Path,
    *,
    known_relation_ids: set[str],
) -> tuple[list[RelationRecord], int]:
    """Convert a reviewed V3 snapshot into additional known-relation training rows."""
    included: list[RelationRecord] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                candidate = TrainingCandidate.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid reviewed snapshot row {line_number}: {exc}"
                ) from exc

            if (
                candidate.relation_id is None
                or candidate.relation_id not in known_relation_ids
                or candidate.relation_name == "UNKNOWN_RELATION"
            ):
                skipped += 1
                continue

            included.append(
                RelationRecord(
                    relation_id=candidate.relation_id,
                    relation_name=candidate.relation_name,
                    relation_description="reviewed production feedback",
                    tokens=candidate.text.split(),
                    marked_text=mark_character_spans(
                        candidate.text,
                        candidate.head,
                        candidate.tail,
                    ),
                    head_text=candidate.head.text,
                    tail_text=candidate.tail.text,
                    source_split="review_snapshot",
                )
            )
    return included, skipped


def accepted_known_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> float:
    confidence = probabilities.max(axis=1)
    mask = confidence >= threshold
    if not mask.any():
        return 0.0
    predictions = probabilities.argmax(axis=1)
    return float(np.mean(predictions[mask] == labels[mask]))


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
    parser.add_argument(
        "--review-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional V3 reviewed examples.jsonl snapshot. Only examples mapped "
            "to existing known relation IDs are added to supervised training."
        ),
    )
    parser.add_argument(
        "--candidate-model-version",
        default="modernbert-open-set-candidate",
        help="Model version written into candidate_report.json for promotion gates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/open_set"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train = read_jsonl(DATA / "train_known.jsonl")
    validation = read_jsonl(DATA / "validation_known.jsonl")
    test_known = read_jsonl(DATA / "test_known.jsonl")
    calibration_unknown = read_jsonl(DATA / "calibration_unknown.jsonl")
    test_unknown = read_jsonl(DATA / "test_unknown.jsonl")

    relation_ids = sorted({record.relation_id for record in train})
    rel2id = {relation_id: index for index, relation_id in enumerate(relation_ids)}
    metadata = {
        str(index): {
            "relation_id": relation_id,
            "relation_name": next(
                record.relation_name
                for record in train
                if record.relation_id == relation_id
            ),
        }
        for relation_id, index in rel2id.items()
    }

    review_included = 0
    review_skipped = 0
    if args.review_snapshot is not None:
        reviewed, review_skipped = load_review_snapshot(
            args.review_snapshot,
            known_relation_ids=set(relation_ids),
        )
        train.extend(reviewed)
        review_included = len(reviewed)
        print(
            json.dumps(
                {
                    "review_snapshot": str(args.review_snapshot),
                    "included_known_examples": review_included,
                    "skipped_unknown_or_unmapped": review_skipped,
                },
                indent=2,
            )
        )

    tokenizer = build_tokenizer(args.model_name)
    model = EntityAwareRelationClassifier(
        args.model_name,
        num_labels=len(rel2id),
        pooling_mode="entity_pair",
        input_view="marked",
    )
    resize_for_markers(model, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    known_collator = KnownCollator(tokenizer, rel2id, args.max_length)
    score_collator = ScoreCollator(tokenizer, args.max_length)

    train_loader = DataLoader(
        RelationDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=known_collator,
    )
    validation_loader = DataLoader(
        RelationDataset(validation),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=known_collator,
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

        for batch in tqdm(train_loader, desc=f"open-set epoch {epoch}/{args.epochs}"):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        validation_logits, validation_labels = logits_and_labels(
            model,
            validation_loader,
            device,
        )
        validation_probabilities = torch.softmax(
            torch.from_numpy(validation_logits),
            dim=-1,
        ).numpy()
        validation_metrics = classification_metrics(
            validation_probabilities,
            validation_labels,
        )
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

    model, tokenizer, _ = EntityAwareRelationClassifier.load_bundle(
        best_dir,
        device=device,
    )
    known_collator = KnownCollator(tokenizer, rel2id, args.max_length)
    score_collator = ScoreCollator(tokenizer, args.max_length)

    validation_logits, validation_labels = logits_and_labels(
        model,
        DataLoader(
            RelationDataset(validation),
            batch_size=args.batch_size * 2,
            collate_fn=known_collator,
        ),
        device,
    )
    calibration_unknown_logits = logits_only(
        model,
        DataLoader(
            RelationDataset(calibration_unknown),
            batch_size=args.batch_size * 2,
            collate_fn=score_collator,
        ),
        device,
    )
    test_known_logits, test_known_labels = logits_and_labels(
        model,
        DataLoader(
            RelationDataset(test_known),
            batch_size=args.batch_size * 2,
            collate_fn=known_collator,
        ),
        device,
    )
    test_unknown_logits = logits_only(
        model,
        DataLoader(
            RelationDataset(test_unknown),
            batch_size=args.batch_size * 2,
            collate_fn=score_collator,
        ),
        device,
    )

    scaler = TemperatureScaler()
    temperature = scaler.fit(validation_logits, validation_labels)

    validation_probabilities = softmax_with_temperature(
        validation_logits,
        temperature,
    )
    calibration_unknown_probabilities = softmax_with_temperature(
        calibration_unknown_logits,
        temperature,
    )
    threshold, calibration_open_set = tune_abstention_threshold(
        validation_probabilities,
        calibration_unknown_probabilities,
    )

    test_known_probabilities = softmax_with_temperature(
        test_known_logits,
        temperature,
    )
    test_unknown_probabilities = softmax_with_temperature(
        test_unknown_logits,
        temperature,
    )

    known_test_metrics = classification_metrics(
        test_known_probabilities,
        test_known_labels,
    )
    unseen_metrics = open_set_metrics(
        test_known_probabilities,
        test_unknown_probabilities,
        threshold=threshold,
    )
    accepted_accuracy = accepted_known_accuracy(
        test_known_probabilities,
        test_known_labels,
        threshold=threshold,
    )

    report = {
        "protocol": "fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown",
        "base_model": args.model_name,
        "seed": args.seed,
        "temperature": temperature,
        "abstention_threshold": threshold,
        "best_validation_macro_f1": best_f1,
        "known_test": known_test_metrics,
        "calibration_open_set": calibration_open_set,
        "unseen_relation_test": unseen_metrics,
        "risk_coverage": risk_coverage_curve(
            test_known_probabilities,
            test_known_labels,
        ),
        "review_snapshot": {
            "path": str(args.review_snapshot) if args.review_snapshot else None,
            "included_known_examples": review_included,
            "skipped_unknown_or_unmapped": review_skipped,
        },
        "history": history,
    }

    candidate_report = {
        "schema_version": "3.0-candidate",
        "model": {
            "model_version": args.candidate_model_version,
            "base_model": args.model_name,
            "pooling_mode": "entity_pair",
            "input_view": "marked",
            "temperature": temperature,
            "default_profile": "balanced",
        },
        "protocol": {
            "name": report["protocol"],
        },
        "known_test": {
            "accuracy": known_test_metrics["accuracy"],
            "macro_f1": known_test_metrics["macro_f1"],
            "micro_f1": known_test_metrics["micro_f1"],
            "top3_accuracy": known_test_metrics["top3_accuracy"],
            "ece": known_test_metrics["ece_15"],
        },
        "final_unseen_test_balanced": {
            "threshold": threshold,
            "auroc": unseen_metrics["ood_auroc"],
            "auprc": unseen_metrics["ood_auprc"],
            "fpr95": unseen_metrics["ood_fpr95"],
            "unknown_f1": unseen_metrics["unknown_f1"],
            "known_acceptance": unseen_metrics["known_acceptance_rate"],
            "unknown_rejection": unseen_metrics["unknown_rejection_rate"],
        },
        "profiles": {
            "balanced": {
                "threshold": threshold,
                "known_acceptance": unseen_metrics["known_acceptance_rate"],
                "accepted_known_accuracy": accepted_accuracy,
                "unknown_rejection": unseen_metrics["unknown_rejection_rate"],
                "unknown_f1": unseen_metrics["unknown_f1"],
            }
        },
        "review_snapshot": report["review_snapshot"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "candidate_report.json").write_text(
        json.dumps(candidate_report, indent=2),
        encoding="utf-8",
    )
    (best_dir / "open_set_config.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "abstention_threshold": threshold,
                "unknown_label": "UNKNOWN_RELATION",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    printable_report = {
        key: value
        for key, value in report.items()
        if key != "risk_coverage"
    }
    print(json.dumps(printable_report, indent=2))


if __name__ == "__main__":
    main()
