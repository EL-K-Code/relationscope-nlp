"""Reproduce the notebook's seven-relation LSTM classification baseline.

This script intentionally follows the ENSAI lab protocol rather than the production
Closed-Set-80 benchmark. Its output must be labeled as "notebook_lineage_7".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from relationscope.data.fewrel import load_closed_set_80
from relationscope.legacy.metrics import legacy_classification_metrics
from relationscope.legacy.notebook_protocol import (
    NOTEBOOK_RELATIONS,
    NOTEBOOK_SEED,
    encode_marked_text,
    notebook_input_vocabulary,
    notebook_split,
)

RAW = Path("data/raw/fewrel")
OUT = Path("artifacts/legacy/lstm_7")


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for notebook reproduction. "
            "Run: uv sync --all-groups --extra legacy"
        ) from exc
    return tf


def _arrays(records, vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    return np.asarray(
        [
            encode_marked_text(
                record.marked_text,
                vocabulary,
                max_length=max_length,
            )
            for record in records
        ],
        dtype=np.int32,
    )


def main() -> None:
    tf = _require_tensorflow()

    np.random.seed(NOTEBOOK_SEED)
    tf.random.set_seed(NOTEBOOK_SEED)

    records = load_closed_set_80(RAW)
    splits = notebook_split(records, NOTEBOOK_RELATIONS, seed=NOTEBOOK_SEED)

    vocabulary = notebook_input_vocabulary(splits["train"])
    max_length = max(len(record.marked_text.split()) for record in splits["train"])

    rel2id = {relation_id: index for index, relation_id in enumerate(NOTEBOOK_RELATIONS)}
    id2rel = {index: relation_id for relation_id, index in rel2id.items()}

    x_train = _arrays(splits["train"], vocabulary, max_length)
    x_validation = _arrays(splits["validation"], vocabulary, max_length)
    x_test = _arrays(splits["test"], vocabulary, max_length)

    y_train = np.asarray(
        [rel2id[record.relation_id] for record in splits["train"]],
        dtype=np.int32,
    )
    y_validation = np.asarray(
        [rel2id[record.relation_id] for record in splits["validation"]],
        dtype=np.int32,
    )
    y_test = np.asarray(
        [rel2id[record.relation_id] for record in splits["test"]],
        dtype=np.int32,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=len(vocabulary),
                output_dim=64,
                mask_zero=True,
            ),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(len(NOTEBOOK_RELATIONS), activation="softmax"),
        ],
        name="notebook_lstm_7",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=10,
        batch_size=32,
        verbose=2,
    )

    validation_probabilities = model.predict(x_validation, verbose=0)
    test_probabilities = model.predict(x_test, verbose=0)

    report = {
        "protocol": "notebook_lineage_7",
        "seed": NOTEBOOK_SEED,
        "relations": list(NOTEBOOK_RELATIONS),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "vocabulary_size": len(vocabulary),
        "max_input_length": max_length,
        "architecture": {
            "embedding_dim": 64,
            "lstm_dim": 128,
            "epochs": 10,
            "batch_size": 32,
        },
        "validation": legacy_classification_metrics(
            validation_probabilities,
            y_validation,
        ),
        "test": legacy_classification_metrics(test_probabilities, y_test),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "id2relation": {str(key): value for key, value in id2rel.items()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "model.keras")
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (OUT / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
