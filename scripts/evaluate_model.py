"""Evaluate a saved Transformer bundle and persist detailed error analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from relationscope.data.io import read_jsonl
from relationscope.models.entity_classifier import (
    EntityAwareRelationClassifier,
    marker_positions,
)
from relationscope.training.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/fewrel_closed_set_80/test.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, label_map = EntityAwareRelationClassifier.load_bundle(
        args.bundle,
        device=device,
    )

    relation_to_index = {
        metadata["relation_id"]: int(index)
        for index, metadata in label_map.items()
    }
    index_to_name = {
        int(index): metadata["relation_name"]
        for index, metadata in label_map.items()
    }

    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []

    model.eval()
    for start in tqdm(range(0, len(records), args.batch_size), desc="evaluate"):
        batch_records = records[start : start + args.batch_size]
        texts = [
            record.marked_text
            if model.input_view == "marked"
            else " ".join(record.tokens)
            for record in batch_records
        ]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if model.pooling_mode == "entity_pair":
            head_pos, tail_pos = marker_positions(encoded["input_ids"], tokenizer)
            model_inputs["head_pos"] = head_pos
            model_inputs["tail_pos"] = tail_pos

        with torch.inference_mode():
            logits = model(**model_inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        probabilities.append(probs)

        for record, row_probs in zip(batch_records, probs, strict=True):
            gold = relation_to_index[record.relation_id]
            predicted = int(np.argmax(row_probs))
            labels.append(gold)
            rows.append(
                {
                    "relation_id_gold": record.relation_id,
                    "relation_name_gold": index_to_name[gold],
                    "relation_id_predicted": label_map[str(predicted)]["relation_id"],
                    "relation_name_predicted": index_to_name[predicted],
                    "confidence": float(row_probs[predicted]),
                    "correct": predicted == gold,
                    "head": record.head_text,
                    "tail": record.tail_text,
                }
            )

    probs_all = np.concatenate(probabilities)
    labels_np = np.asarray(labels)
    predictions_np = probs_all.argmax(axis=1)

    report = classification_report(
        labels_np,
        predictions_np,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels_np, predictions_np)

    confusions: Counter[tuple[int, int]] = Counter()
    for gold, predicted in zip(labels_np.tolist(), predictions_np.tolist(), strict=True):
        if gold != predicted:
            confusions[(gold, predicted)] += 1

    top_confusions = [
        {
            "gold": index_to_name[gold],
            "predicted": index_to_name[predicted],
            "count": count,
        }
        for (gold, predicted), count in confusions.most_common(30)
    ]

    confidence_buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        confidence = float(row["confidence"])
        bucket_start = min(int(confidence * 10), 9) / 10
        key = f"{bucket_start:.1f}-{bucket_start + 0.1:.1f}"
        confidence_buckets[key][0] += 1
        confidence_buckets[key][1] += int(bool(row["correct"]))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "bundle": str(args.bundle),
                "dataset": str(args.dataset),
                "metrics": classification_metrics(probs_all, labels_np),
                "classification_report": report,
                "top_confusions": top_confusions,
                "confidence_buckets": {
                    key: {
                        "count": values[0],
                        "accuracy": values[1] / values[0],
                    }
                    for key, values in confidence_buckets.items()
                },
                "confusion_matrix": matrix.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(classification_metrics(probs_all, labels_np), indent=2))


if __name__ == "__main__":
    main()
