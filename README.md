# RelationScope NLP

**Production-grade Relation Extraction & Knowledge Structuring Engine**

RelationScope turns text + an explicit entity pair into a ranked relation prediction, calibrated confidence, optional `UNKNOWN_RELATION` abstention, machine-readable triples and token-level evidence.

The project is the engineering continuation of an ENSAI Advanced NLP lab by **Maxime Ferret and Alex Komla Labou**. The original experimental path is preserved instead of being hidden behind a modern Transformer.

> **Current status:** the full software and experiment scaffolding is implemented. Model-training and benchmark runs are intentionally pending. No new portfolio metric is claimed before reproduction.

## Research story

```text
ENSAI notebook
     |
     +--> LSTM classification
     |
     +--> Seq2Seq + Attention
     |
     +--> T5 generation
     |
     +--> K=7 -> 30 scaling study
                |
                v
      controlled marker ablation
                |
                v
      entity-aware Transformer
                |
                v
      calibration + abstention
                |
                v
      unseen-relation detection
                |
                v
      FastAPI + batch inference
                |
                v
      Next.js product interface
                |
                v
      private API on Cloud Run
```

## What is implemented

### Notebook lineage

Committed reproduction scripts preserve:

- the original seven-relation LSTM;
- the original 70/30 then 80/20 split logic;
- Seq2Seq LSTM + attention;
- T5 generation;
- legacy 32-token T5 target evaluation and corrected non-truncating evaluation;
- the K = 7, 10, 15, 20, 30 LSTM/BiLSTM scaling study.

See:

- `docs/notebook_lineage.md`
- `docs/notebook_reference_metrics.md`

Historical notebook metrics are stored only as reproduction targets, not as final RelationScope claims.

### Production NLP track

Implemented:

- FewRel Closed-Set-80 protocol;
- word + character TF-IDF logistic-regression baseline;
- ModernBERT-compatible Transformer classifier;
- HEAD / TAIL special tokens;
- CLS-only and entity-aware pooling modes;
- controlled entity-marker ablation;
- Accuracy, Macro-F1, Micro-F1, Top-3 Accuracy and ECE;
- temperature scaling;
- risk-coverage analysis;
- relation-level open-set benchmark;
- `UNKNOWN_RELATION` abstention;
- OOD AUROC / AUPRC / FPR95 / Unknown-F1;
- token-occlusion evidence;
- detailed error-analysis export;
- latency benchmark;
- generated model card;
- explicit model-promotion gates.

### Product/API track

FastAPI endpoints:

```text
GET  /health
GET  /ready
GET  /v1/model

POST /v1/relations:predict
POST /v1/relations:predict-batch
POST /v1/relations:predict-open-set
POST /v1/relations:explain
```

The API supports:

- single inference;
- vectorized batch inference;
- top-k relations;
- calibrated open-set decisions;
- token-occlusion evidence;
- privacy-preserving structured request logging.

### Frontend

`apps/web` contains a Next.js App Router interface with:

- sentence + HEAD/TAIL entity inputs;
- top-k confidence visualization;
- KNOWN vs UNKNOWN decision card;
- evidence heatmap;
- CSV batch import;
- downloadable sample CSV;
- batch result export;
- extracted knowledge graph;
- Model Lab explaining the research progression.

### Deployment

Implemented deployment architecture:

```text
Browser
   |
   v
Public Next.js Cloud Run
   |
   | Google-signed identity token
   v
Private FastAPI Cloud Run
   |
   | read-only IAM
   v
Versioned GCS model bundle
```

Also included:

- separate API and web Dockerfiles;
- Docker Compose for local integration;
- Terraform for Cloud Run / GCS / Artifact Registry / IAM;
- keyless GitHub -> GCP deployment with Workload Identity Federation;
- SHA-256 model artifact manifests;
- verified model download from GCS;
- CI for Python and Next.js.

## Core model architecture

Default production classifier:

```text
sentence
  +
[HEAD] entity [/HEAD]
[TAIL] entity [/TAIL]
       |
       v
Transformer encoder
       |
       +---- CLS
       +---- HEAD representation
       +---- TAIL representation
       +---- abs(HEAD - TAIL)
                   |
                   v
             concatenation
                   |
                   v
            relation logits
                   |
             temperature
              scaling
                   |
                   v
         relation / abstain
```

## Entity-marker ablation

The project does not assume that entity markers help. It measures the effect under matched conditions:

| Condition | Input | Pooling |
|---|---|---|
| A | raw text | CLS |
| B | HEAD/TAIL markers | CLS |
| C | HEAD/TAIL markers | CLS + HEAD + TAIL + abs(HEAD-TAIL) |

The result is not filled in until the experiments are actually run.

## Open-set protocol

The open-set extension keeps relation identities separated:

```text
56 official-train relations
    -> classifier training

8 different official-train relations
    -> threshold calibration only

16 official-validation relations
    -> untouched final unseen-relation test
```

This prevents tuning the rejection threshold on the final unseen-relation test set.

## Repository layout

```text
.
├── apps/
│   └── web/                        # Next.js recruiter/product interface
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── api.md
│   ├── gcp_deployment.md
│   ├── local_development.md
│   ├── methodology.md
│   ├── model_promotion.md
│   ├── notebook_lineage.md
│   ├── notebook_reference_metrics.md
│   └── premium_experiments.md
├── infra/
│   └── terraform/
├── reports/
├── scripts/
│   ├── download_fewrel.py
│   ├── prepare_fewrel.py
│   ├── reproduce_notebook_lstm.py
│   ├── reproduce_notebook_attention.py
│   ├── reproduce_notebook_t5.py
│   ├── reproduce_notebook_scaling.py
│   ├── train_tfidf_baseline.py
│   ├── train_transformer.py
│   ├── run_marker_ablation.py
│   ├── prepare_open_set_fewrel.py
│   ├── train_open_set_transformer.py
│   ├── evaluate_model.py
│   ├── benchmark_latency.py
│   ├── build_model_card.py
│   └── publish_model_bundle.py
├── src/
│   └── relationscope/
│       ├── api/
│       ├── data/
│       ├── inference/
│       ├── legacy/
│       ├── models/
│       └── training/
├── tests/
├── compose.yaml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Installation

Python 3.11-3.13 is supported. Python 3.12 is recommended.

```bash
uv sync --all-groups
```

To reproduce the TensorFlow notebook lineage later:

```bash
uv sync --all-groups --extra legacy
```

Frontend dependencies:

```bash
cd apps/web
npm install
```

## Execution order when experiments begin

No need to run this yet. The planned order is frozen as:

```text
1. tests + data integrity
2. notebook LSTM reproduction
3. notebook attention reproduction
4. notebook T5 reproduction/correction
5. notebook K-scaling reproduction
6. TF-IDF baseline
7. closed-set Transformer
8. marker ablation
9. open-set Transformer
10. calibration / error analysis
11. latency benchmark
12. model promotion
13. GCS publication
14. Cloud Run deployment
```

## Methodological guardrails

- FewRel Closed-Set-80 is **not** presented as the official FewRel few-shot leaderboard protocol.
- The legacy T5 32-token result is separated from full-target exact match.
- Final unseen relations are not used for threshold tuning.
- Metrics enter the README/CV only after committed scripts reproduce them.
- Raw sentences are not written to production request logs.
- GCP is the serving/industrialization layer; NLP experimentation remains the core of the project.

## Data and attribution

FewRel is provided by THUNLP. If using the dataset, cite:

> Han et al. (2018), *FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation*, EMNLP 2018.

Official repository: https://github.com/thunlp/FewRel

Raw datasets, trained weights and credentials are not committed to Git.
