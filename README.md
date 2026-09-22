# RelationScope NLP

**Production-grade Relation Extraction & Knowledge Structuring Engine**

RelationScope turns entity-marked natural-language text into calibrated relation predictions and machine-readable knowledge triples.

The project grows out of an ENSAI Advanced NLP lab on FewRel (LSTM, attention, T5), but the production code is a clean-room rewrite with a corrected evaluation protocol, reproducible baselines, tests, an API, and a path to GCP deployment.

## What the system does

Input:

```text
Marie Curie was born in Warsaw in 1867.
HEAD = Marie Curie
TAIL = Warsaw
```

Output:

```json
{
  "head": "Marie Curie",
  "relation": "place of birth",
  "tail": "Warsaw",
  "confidence": 0.964
}
```

Core tasks:

- relation classification;
- explicit head/tail entity marking;
- entity-aware Transformer representations;
- top-k calibrated predictions;
- batch knowledge-triple extraction;
- later: abstention / unknown-relation detection.

## Benchmark protocol

FewRel 1.0 officially uses **64 training relations + 16 disjoint validation relations** for few-shot evaluation. RelationScope does **not** claim leaderboard comparability with that protocol.

For the production classifier we explicitly define a separate benchmark:

> **FewRel Closed-Set-80** = the 64 train relations and 16 validation relations combined, then split *within each relation* into deterministic train/validation/test partitions.

This answers a different question: *how well can a deployed classifier recognize a fixed catalogue of 80 known relations?*

All reports must state the protocol name.

## V1 architecture

```text
FewRel JSON
    |
    v
Entity span validation
    |
    v
[HEAD] ... [/HEAD] / [TAIL] ... [/TAIL]
    |
    +--------------------------+
    |                          |
    v                          v
TF-IDF + Logistic       Transformer encoder
Regression baseline     + entity-aware pooling
    |                          |
    +------------+-------------+
                 v
       Accuracy / Macro-F1
       Top-k / ECE / latency
                 |
                 v
             FastAPI
```

## Repository layout

```text
src/relationscope/
  api/            FastAPI service
  data/           FewRel parsing, entity markers, splits
  inference/      production predictor
  models/         entity-aware Transformer classifier
  training/       metrics and training utilities

scripts/
  download_fewrel.py
  prepare_fewrel.py
  train_tfidf_baseline.py
  train_transformer.py

tests/
docs/
```

## Quick start

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EL-K-Code/relationscope-nlp.git
cd relationscope-nlp

uv sync --all-groups
```

Download the official FewRel files:

```bash
uv run python scripts/download_fewrel.py
```

Prepare deterministic Closed-Set-80 splits:

```bash
uv run python scripts/prepare_fewrel.py
```

Train the lexical baseline:

```bash
uv run python scripts/train_tfidf_baseline.py
```

Train the entity-aware Transformer:

```bash
uv run python scripts/train_transformer.py \
  --model-name answerdotai/ModernBERT-base \
  --epochs 3
```

Run tests:

```bash
uv run pytest
uv run ruff check .
```

## API

After a model bundle exists:

```bash
RELATIONSCOPE_MODEL_DIR=artifacts/transformer/best \
uv run uvicorn relationscope.api.main:app --host 0.0.0.0 --port 8080
```

Example request:

```json
{
  "text": "Marie Curie was born in Warsaw in 1867.",
  "head": {"text": "Marie Curie", "start": 0, "end": 11},
  "tail": {"text": "Warsaw", "start": 24, "end": 30},
  "top_k": 3
}
```

## Roadmap

- **V1** — reproducible Closed-Set-80 benchmark + TF-IDF baseline + entity-aware Transformer.
- **V2** — model comparison, calibration, error analysis, experiment tracking.
- **V3** — React/Next.js UI + batch extraction + knowledge graph view.
- **V4** — Docker, Cloud Run, Artifact Registry, CI/CD, monitoring.
- **V5** — unknown-relation detection, abstention, active learning.

## Data and attribution

FewRel is provided by THUNLP and the official repository is MIT-licensed. If you use the dataset, cite:

> Han et al. (2018), *FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation*, EMNLP 2018.

Official repository: https://github.com/thunlp/FewRel

Raw data are downloaded locally and are not committed to this repository.
