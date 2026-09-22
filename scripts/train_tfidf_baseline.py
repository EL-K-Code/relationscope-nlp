"""Train a strong lexical baseline before any Transformer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from relationscope.data.io import read_jsonl
from relationscope.training.metrics import classification_metrics

DATA = Path("data/processed/fewrel_closed_set_80")
OUT = Path("artifacts/tfidf")


def main() -> None:
    train = read_jsonl(DATA / "train.jsonl")
    validation = read_jsonl(DATA / "validation.jsonl")
    test = read_jsonl(DATA / "test.jsonl")

    relation_ids = sorted({record.relation_id for record in train})
    rel2id = {relation_id: index for index, relation_id in enumerate(relation_ids)}

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", features),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    max_iter=100,
                    tol=1e-4,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    average=True,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(
        [record.marked_text for record in train],
        [rel2id[record.relation_id] for record in train],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT / "model.joblib")

    report: dict[str, object] = {
        "protocol": "fewrel_closed_set_80",
        "model": "tfidf_word_char_sgd_logistic_regression",
    }
    for split_name, records in (("validation", validation), ("test", test)):
        labels = np.array([rel2id[record.relation_id] for record in records])
        probabilities = pipeline.predict_proba([record.marked_text for record in records])
        report[split_name] = classification_metrics(probabilities, labels)

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
