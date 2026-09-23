# RelationScope NLP

**Production-grade relation extraction with calibrated open-set abstention**

RelationScope turns a sentence plus an explicit HEAD/TAIL entity pair into ranked relation predictions, calibrated confidence, and a defensible `UNKNOWN_RELATION` decision when the model is not confident enough.

The project extends an ENSAI Advanced NLP lab by **Maxime Ferret and Alex Komla Labou** into a reproducible ML system with controlled ablations, open-set evaluation, FastAPI serving, a Next.js product interface, model-bundle integrity checks, Terraform infrastructure, and deployment on Google Cloud Run.

> **Status: v1.0.0 production release candidate.**  
> The promoted model is deployed and serving live inference through a public Next.js frontend backed by a private Cloud Run API.

## Highlights

- **ModernBERT-base** with explicit HEAD/TAIL entity markers.
- **Entity-aware pooling:** CLS + HEAD + TAIL + |HEAD − TAIL|.
- **56 known relations** with **16 completely unseen final-test relations**.
- Temperature-scaled confidence with three frozen operating profiles.
- `UNKNOWN_RELATION` abstention for open-set inference.
- FastAPI private backend + public Next.js frontend.
- Model bundle stored in GCS with SHA-256 manifest verification.
- Terraform-managed Cloud Run, IAM, Artifact Registry, and GCS.
- CPU production inference on Cloud Run with scale-to-zero.

## Production model

Promoted bundle:

```text
modernbert-open-set-v1.1
```

Protocol:

```text
fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown
```

Split sizes:

| Split | Examples |
|---|---:|
| train_known | 31,360 |
| validation_known | 3,920 |
| test_known | 3,920 |
| calibration_unknown | 5,600 |
| test_unknown | 11,200 |

Relation allocation:

- 56 known relations for classifier training/evaluation.
- 8 different relations used only for unknown-threshold calibration.
- 16 untouched relations used as the final unseen-relation test.

The final unseen relations are not used to tune the operating threshold.

## Final evaluation

### Known-relation test set

| Metric | Result |
|---|---:|
| Accuracy | **89.21%** |
| Macro-F1 | **89.16%** |
| Micro-F1 | **89.21%** |
| Top-3 accuracy | **98.01%** |
| ECE | **1.95%** |

### Final unseen-relation test

| Metric | Result |
|---|---:|
| OOD AUROC | **80.33%** |
| OOD AUPRC | **90.50%** |
| FPR95 | **57.86%** |
| Unknown-F1 | **85.30%** |
| Known acceptance | **61.25%** |
| Unknown rejection | **84.45%** |

These final-test open-set numbers use the **balanced threshold 0.967**, frozen from validation-known + calibration-unknown data.

## Operating profiles

Thresholds were selected before evaluation on the final unseen-relation test set.

| Profile | Threshold | Known coverage | Accuracy on accepted known | Unknown rejection | Unknown-F1 |
|---|---:|---:|---:|---:|---:|
| High coverage | 0.924 | 72.40% | 97.67% | 73.79% | 80.44% |
| **Balanced** | **0.967** | **61.25%** | **98.67%** | **84.45%** | **85.30%** |
| High safety | 0.978 | 55.43% | 98.94% | 88.63% | 86.80% |

`balanced` is the production default.

## Selective prediction

Known-test risk decreases strongly as the model abstains on lower-confidence examples:

| Coverage | Error risk | Accuracy |
|---:|---:|---:|
| 100% | 10.8% | 89.2% |
| 90% | 6.2% | 93.8% |
| 80% | 3.4% | 96.6% |
| 70% | 1.9% | 98.1% |
| 60% | 1.3% | 98.7% |
| 50% | 0.9% | 99.1% |

This curve concerns selective classification among known examples only.

## Latency

### GPU benchmark

Sequential open-set inference on GPU:

- mean: **22.04 ms**
- median: **21.81 ms**
- P95: **23.32 ms**
- P99: **28.49 ms**
- throughput: **45.37 examples/s**

Batch-8 reached approximately **191 examples/s**.

### Cloud Run CPU — warm

Production Cloud Run configuration: **2 vCPU / 4 GiB**.

- application inference mean: **94.2 ms**
- application inference median: **87.2 ms**
- end-to-end client mean: **157.3 ms**
- end-to-end client median: **138.0 ms**

The CPU benchmark is intentionally reported separately from the GPU benchmark. Scale-to-zero cold start is not included in the warm figures above.

## Entity-information ablation

ModernBERT-base was evaluated under matched conditions on the Closed-Set-80 protocol:

| Condition | Input | Pooling | Test Accuracy | Test Macro-F1 | Top-3 |
|---|---|---|---:|---:|---:|
| A | raw | CLS | 63.98% | 63.02% | 86.73% |
| B | marked | CLS | 79.18% | 78.84% | 93.13% |
| C | marked | entity-pair | **85.02%** | **84.89%** | **95.96%** |

Effects on test Macro-F1:

- entity markers: **+15.82 percentage points**
- entity-aware pooling beyond markers: **+6.05 percentage points**
- full structured entity information vs raw CLS: **+21.87 percentage points**

## System architecture

```text
Browser
   |
   v
Public Next.js service
Google Cloud Run
   |
   | Google-signed identity token
   v
Private FastAPI service
Google Cloud Run
   |
   | read-only IAM
   v
Versioned GCS model bundle
   |
   v
ModernBERT + temperature scaling
   |
   v
Relation prediction / UNKNOWN_RELATION
```

The browser never receives Google Cloud credentials. The Next.js server-side proxy obtains an identity token using its Cloud Run service account and invokes the private API.

## API

```text
GET  /health
GET  /ready
GET  /v1/model

POST /v1/relations:predict
POST /v1/relations:predict-batch
POST /v1/relations:predict-open-set
POST /v1/relations:explain
```

Example open-set response:

```json
{
  "model_id": "modernbert-open-set-v1.1",
  "decision": "UNKNOWN_RELATION",
  "accepted": false,
  "profile": "balanced",
  "threshold": 0.967
}
```

## Frontend

The Next.js interface includes:

- sentence + HEAD/TAIL entity inputs;
- High coverage / Balanced / High safety profile selector;
- top-k ranked relations;
- confidence, threshold, and decision margin;
- KNOWN vs UNKNOWN decision card;
- token-occlusion evidence;
- CSV batch inference and export;
- extracted knowledge graph;
- production KPI cards;
- risk–coverage visualization;
- frozen evaluation protocol and operating-point dashboard.

## Model-bundle integrity

The promoted model lives outside Git and is published to GCS.

At runtime the service:

1. downloads the configured model prefix;
2. verifies the bundle against a SHA-256 manifest;
3. atomically promotes the downloaded directory;
4. loads the encoder, classifier, tokenizer, calibration settings, and relation mapping.

Current promoted version:

```text
modernbert-open-set-v1.1
```

## Research lineage

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
      ModernBERT + entity-aware pooling
                |
                v
      temperature scaling
                |
                v
      unseen-relation detection
                |
                v
      selective prediction
                |
                v
      FastAPI + Next.js
                |
                v
      Google Cloud production deployment
```

## Repository layout

```text
.
├── apps/web/                     # Next.js application
├── docs/                         # Methodology and deployment documentation
├── infra/terraform/              # GCP infrastructure as code
├── reports/                      # Frozen evaluation reports
├── scripts/                      # Reproduction, training and publication scripts
├── src/relationscope/
│   ├── api/
│   ├── data/
│   ├── inference/
│   ├── legacy/
│   ├── models/
│   └── training/
├── tests/
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

## Local development

Python 3.12 is recommended.

```bash
uv sync --all-groups
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```

Local environment variables are loaded from `.env` without overriding environment variables already supplied by the process.

## Deployment

Infrastructure is managed with Terraform. Production containers are built in Google Cloud Build and stored in Artifact Registry.

The production model bucket is versioned, the API service account has read-only access to model objects, the frontend is public, and the API remains private to authorized callers.

See:

- `docs/gcp_deployment.md`
- `infra/terraform/`

## Methodological guardrails

- Closed-Set-80 is not presented as the official FewRel few-shot leaderboard protocol.
- Final unseen relations are never used for threshold selection.
- Calibration metrics and final-test metrics are reported separately.
- AUPRC is not compared across splits without considering class prevalence.
- GPU and Cloud Run CPU latency are labeled separately.
- Token occlusion is presented as evidence-by-occlusion, not as a causal explanation.
- Raw sentences are not committed with model artifacts or infrastructure state.

## Data and attribution

FewRel is provided by THUNLP. If using the dataset, cite:

> Han et al. (2018), *FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation*, EMNLP 2018.

Official repository: https://github.com/thunlp/FewRel

Raw datasets, trained weights, Terraform variable files, and credentials are not committed to Git.
