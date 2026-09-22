"""Reproduce the ENSAI notebook's LSTM encoder-decoder with attention.

The goal is lineage and interpretability, not the production serving model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from relationscope.data.fewrel import load_closed_set_80
from relationscope.legacy.notebook_protocol import (
    NOTEBOOK_RELATIONS,
    NOTEBOOK_SEED,
    notebook_label,
    notebook_split,
)

RAW = Path("data/raw/fewrel")
OUT = Path("artifacts/legacy/seq2seq_attention")


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for notebook reproduction. "
            "Run: uv sync --all-groups --extra legacy"
        ) from exc
    return tf


def clean_relation_description(text: str) -> str:
    """Reproduce the notebook's relation-label normalization."""
    text = text.lower()
    text = re.sub(r"\([^)]*\bp\d+\b[^)]*\)", " ", text)
    text = re.sub(r"\([^)]*property:p\d+[^)]*\)", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_vocabularies(train_records):
    input_counter: Counter[str] = Counter()
    for record in train_records:
        input_counter.update(token.lower() for token in record.marked_text.split())

    input_vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in input_counter.most_common(5000):
        if token not in input_vocab:
            input_vocab[token] = len(input_vocab)

    relation_labels = [notebook_label(record) for record in train_records]
    output_counter: Counter[str] = Counter()
    for label in relation_labels:
        output_counter.update(clean_relation_description(label).split())

    output_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for token in output_counter:
        if token not in output_vocab:
            output_vocab[token] = len(output_vocab)

    return input_vocab, output_vocab


def encode_input(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    tokens = text.lower().split()
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    ids = ids[:max_length]
    return ids + [vocab["<pad>"]] * (max_length - len(ids))


def encode_output(label: str, vocab: dict[str, int], max_length: int) -> list[int]:
    tokens = clean_relation_description(label).split()
    ids = [
        vocab["<bos>"],
        *[vocab.get(token, vocab["<unk>"]) for token in tokens],
        vocab["<eos>"],
    ]
    ids = ids[:max_length]
    return ids + [vocab["<pad>"]] * (max_length - len(ids))


def build_arrays(records, input_vocab, output_vocab, max_input, max_output):
    x = np.asarray(
        [
            encode_input(record.marked_text, input_vocab, max_input)
            for record in records
        ],
        dtype=np.int32,
    )
    y = np.asarray(
        [
            encode_output(notebook_label(record), output_vocab, max_output)
            for record in records
        ],
        dtype=np.int32,
    )
    return x, y


def shift_right(y: np.ndarray) -> np.ndarray:
    return np.pad(y[:, :-1], ((0, 0), (0, 1)), constant_values=0)


def shift_left(y: np.ndarray) -> np.ndarray:
    return np.pad(y[:, 1:], ((0, 0), (0, 1)), constant_values=0)


def main() -> None:
    tf = _require_tensorflow()

    np.random.seed(NOTEBOOK_SEED)
    tf.random.set_seed(NOTEBOOK_SEED)

    records = load_closed_set_80(RAW)
    splits = notebook_split(records, NOTEBOOK_RELATIONS, seed=NOTEBOOK_SEED)

    input_vocab, output_vocab = build_vocabularies(splits["train"])
    inverse_output = {value: key for key, value in output_vocab.items()}

    max_input = max(len(record.marked_text.split()) for record in splits["train"])
    max_output = max(
        len(clean_relation_description(notebook_label(record)).split())
        for record in splits["train"]
    ) + 2

    x_train, y_train = build_arrays(
        splits["train"], input_vocab, output_vocab, max_input, max_output
    )
    x_validation, y_validation = build_arrays(
        splits["validation"], input_vocab, output_vocab, max_input, max_output
    )
    x_test, y_test = build_arrays(
        splits["test"], input_vocab, output_vocab, max_input, max_output
    )

    embedding_dim = 64
    lstm_dim = 128

    encoder_inputs = tf.keras.Input(shape=(max_input,), name="encoder_input")
    encoder_embedding = tf.keras.layers.Embedding(
        len(input_vocab),
        embedding_dim,
        mask_zero=False,
        name="encoder_embedding",
    )(encoder_inputs)
    encoder_outputs, encoder_h, encoder_c = tf.keras.layers.LSTM(
        lstm_dim,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )(encoder_embedding)

    decoder_inputs = tf.keras.Input(shape=(max_output,), name="decoder_input")
    decoder_embedding = tf.keras.layers.Embedding(
        len(output_vocab),
        embedding_dim,
        mask_zero=False,
        name="decoder_embedding",
    )(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        lstm_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[encoder_h, encoder_c],
    )

    attention = tf.keras.layers.Attention(name="attention")
    context = attention([decoder_outputs, encoder_outputs])
    concatenated = tf.keras.layers.Concatenate(name="context_concat")(
        [decoder_outputs, context]
    )
    projection = tf.keras.layers.Dense(
        len(output_vocab),
        activation="softmax",
        name="output_projection",
    )
    outputs = projection(concatenated)

    model = tf.keras.Model(
        [encoder_inputs, decoder_inputs],
        outputs,
        name="seq2seq_with_attention",
    )

    def masked_loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

    model.compile(
        optimizer="adam",
        loss=masked_loss,
        metrics=["accuracy"],
    )

    history = model.fit(
        [x_train, shift_right(y_train)],
        shift_left(y_train),
        validation_data=(
            [x_validation, shift_right(y_validation)],
            shift_left(y_validation),
        ),
        epochs=15,
        batch_size=32,
        verbose=2,
    )

    test_loss, teacher_forced_accuracy = model.evaluate(
        [x_test, shift_right(y_test)],
        shift_left(y_test),
        batch_size=32,
        verbose=0,
    )

    def decode_relation(x: np.ndarray) -> str:
        generated = [output_vocab["<bos>"]]
        for _ in range(max_output):
            padded = generated + [output_vocab["<pad>"]] * (max_output - len(generated))
            predictions = model.predict(
                [x, np.asarray([padded], dtype=np.int32)],
                verbose=0,
            )
            next_id = int(np.argmax(predictions[0, len(generated) - 1]))
            if next_id == output_vocab["<eos>"]:
                break
            generated.append(next_id)

        return " ".join(
            inverse_output[index]
            for index in generated[1:]
            if index not in {
                output_vocab["<pad>"],
                output_vocab["<bos>"],
                output_vocab["<eos>"],
            }
        )

    correct = 0
    examples = []
    for index, record in enumerate(splits["test"]):
        prediction = decode_relation(x_test[index : index + 1])
        gold = clean_relation_description(notebook_label(record))
        is_correct = prediction == gold
        correct += int(is_correct)

        if len(examples) < 50:
            examples.append(
                {
                    "text": record.marked_text,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                }
            )

    autoregressive_exact_match = correct / max(len(splits["test"]), 1)

    report = {
        "protocol": "notebook_seq2seq_attention_7",
        "seed": NOTEBOOK_SEED,
        "relations": list(NOTEBOOK_RELATIONS),
        "max_input_length": max_input,
        "max_output_length": max_output,
        "input_vocab_size": len(input_vocab),
        "output_vocab_size": len(output_vocab),
        "teacher_forced_test_loss": float(test_loss),
        "teacher_forced_test_accuracy": float(teacher_forced_accuracy),
        "autoregressive_exact_match": float(autoregressive_exact_match),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "sample_predictions": examples,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "model.keras")
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
