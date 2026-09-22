"""Reproduce the notebook's relation-cardinality scaling study.

The script intentionally reuses the seven-relation vocabulary and maximum input
length, matching the original notebook's global-variable behavior.
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
    SCALING_K_VALUES,
    encode_marked_text,
    notebook_input_vocabulary,
    notebook_split,
    top_k_relations,
)

RAW = Path("data/raw/fewrel")
OUT = Path("artifacts/legacy/scaling")


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


def _baseline_model(tf, vocab_size: int, n_relations: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(n_relations, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _bilstm_dropout_model(tf, vocab_size: int, n_relations: int):
    inputs = tf.keras.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True)(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.0,
        )
    )(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_relations, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _evaluate_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
) -> dict[str, object]:
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=10,
        batch_size=32,
        verbose=0,
    )
    validation_probabilities = model.predict(x_validation, verbose=0)
    test_probabilities = model.predict(x_test, verbose=0)

    return {
        "validation": legacy_classification_metrics(
            validation_probabilities,
            y_validation,
        ),
        "test": legacy_classification_metrics(test_probabilities, y_test),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }


def main() -> None:
    tf = _require_tensorflow()
    records = load_closed_set_80(RAW)

    initial = notebook_split(records, NOTEBOOK_RELATIONS, seed=NOTEBOOK_SEED)
    vocabulary = notebook_input_vocabulary(initial["train"])
    max_length = max(len(record.marked_text.split()) for record in initial["train"])

    results: list[dict[str, object]] = []

    for k in SCALING_K_VALUES:
        tf.keras.backend.clear_session()
        np.random.seed(NOTEBOOK_SEED)
        tf.random.set_seed(NOTEBOOK_SEED)

        selected = top_k_relations(records, k)
        splits = notebook_split(records, selected, seed=NOTEBOOK_SEED)
        rel2id = {relation_id: index for index, relation_id in enumerate(selected)}

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

        baseline = _baseline_model(tf, len(vocabulary), k)
        baseline_result = _evaluate_model(
            baseline,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
        )

        tf.keras.backend.clear_session()
        np.random.seed(NOTEBOOK_SEED)
        tf.random.set_seed(NOTEBOOK_SEED)

        bilstm = _bilstm_dropout_model(tf, len(vocabulary), k)
        bilstm_result = _evaluate_model(
            bilstm,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
        )

        row = {
            "k": k,
            "relations": selected,
            "split_sizes": {name: len(items) for name, items in splits.items()},
            "baseline_lstm": baseline_result,
            "bilstm_dropout": bilstm_result,
        }
        results.append(row)

        print(
            f"K={k:>2} | "
            f"baseline val_acc={baseline_result['validation']['accuracy']:.3f} | "
            f"BiLSTM val_acc={bilstm_result['validation']['accuracy']:.3f}"
        )

    report = {
        "protocol": "notebook_scaling_k",
        "seed": NOTEBOOK_SEED,
        "k_values": list(SCALING_K_VALUES),
        "shared_initial_7_relation_vocabulary": True,
        "vocabulary_size": len(vocabulary),
        "max_input_length": max_length,
        "results": results,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
