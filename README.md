# RelationScope NLP

**Document relation intelligence with open-set inference and governed continuous learning**

RelationScope turns a sentence plus an explicit HEAD/TAIL entity pair into ranked relation predictions, calibrated confidence, and a defensible `UNKNOWN_RELATION` decision when the model is not confident enough.

The project extends an ENSAI Advanced NLP lab by **Maxime Ferret and Alex Komla Labou** into a reproducible ML product: ModernBERT relation classification, controlled entity-marker ablations, open-set evaluation, word-level prediction influence, human feedback, durable batch inference, model lifecycle management, FastAPI/Next.js serving, Terraform, and Google Cloud.

> **Current production checkpoint:** `v1.0.0` with `modernbert-open-set-v1.1`.  
> **Current development line:** V3 adds automatic document intelligence, active learning, drift detection, versioned retraining data, promotion gates, champion/challenger canaries and rollback around the frozen benchmark.

## What V2 adds

### Word influence instead of attention-as-explanation

The original ENSAI sequence-to-sequence experiment could visualize decoder attention. The promoted production system is a ModernBERT classifier with entity-aware pooling, so V2 uses **relation-specific word occlusion** instead of claiming that Transformer self-attention directly explains a relation.

For each complete displayed word, V2 masks all of its tokenizer pieces and measures the change in calibrated probability of the target relation:

```text
influence(word) =
P(target relation | original)
-
P(target relation | word masked)
```

The UI renders sentence-order evidence:

- supporting words;
- opposing words;
- strongest supporting cues;
- signed confidence effect.

This is perturbation-based evidence, not a causal explanation.

### Human review loop

After inference a reviewer can:

- confirm the prediction;
- correct the relation;
- mark the example unknown.

Prediction metadata is stored without raw text by default. The original sentence and entity spans are persisted only after explicit training-data consent.

Consented feedback can be exported for offline review/retraining; it is never used for automatic self-training.

### Privacy-preserving observability

V2 records operational metadata:

- model version;
- operating profile;
- KNOWN/UNKNOWN decision;
- confidence and threshold;
- inference latency;
- input length and SHA-256 digest.

Raw sentence text is not stored in telemetry.

The Ops dashboard reports recent prediction count, KNOWN/UNKNOWN rate, confidence, p50/p95 latency, model versions and profile usage.

### Durable asynchronous batch jobs

Large CSV extraction requests become durable jobs:

```text
Next.js
  ↓
private FastAPI
  ↓
GCS job state
  ↓
Cloud Tasks
  ↓ OIDC
private worker callback
  ↓
vectorized ModernBERT chunks
  ↓
durable results
```

The public proxy blocks internal worker routes.

### Model registry and promotion

V2 separates immutable registration from production promotion:

```text
training output
      ↓
models/releases/<model-version>
      ↓ explicit promotion
models/current
      ↓
production
```

Promotion is auditable and model artifacts remain SHA-256 verified.

See **[`docs/v2.md`](docs/v2.md)** for the complete design.

## What V3 adds

V3 completes the planned RelationScope roadmap:

- raw text / PDF / TXT document analysis without manually supplied HEAD/TAIL spans;
- automatic entity discovery with a deterministic offline fallback and optional transformer NER;
- sentence-aware, bounded candidate-pair generation;
- document-level knowledge graph assembly;
- uncertainty-driven active-learning review queue;
- explicit consent before raw review examples are persisted;
- versioned reviewed datasets with SHA-256 manifests;
- reviewed-data augmentation in the existing open-set training pipeline;
- champion/challenger promotion gates on the frozen evaluation protocol;
- production drift signals from confidence, UNKNOWN rate, text length and relation distribution;
- optional private challenger Cloud Run service;
- sticky deterministic canary routing in the public Next.js proxy;
- audited model rollback.

The intended lifecycle is:

```text
production
   ↓
telemetry + uncertain examples
   ↓
human review
   ↓
versioned dataset
   ↓
candidate retraining
   ↓
frozen evaluation
   ↓
promotion gates
   ↓
challenger canary
   ↓
promote / rollback
```

V3 does not introduce automatic self-training. Human review and explicit promotion remain required.

See [`docs/v3.md`](docs/v3.md) for the complete design.

## Production model

Promoted bundle:

```text
modernbert-open-set-v1.1
```

Architecture:

```text
marked sentence
      ↓
ModernBERT-base
      ↓
CLS + HEAD + TAIL + |HEAD - TAIL|
      ↓
relation logits
      ↓
temperature scaling
      ↓
profile threshold
      ↓
KNOWN relation / UNKNOWN_RELATION
```

Protocol:

```text
fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown
```

| Split | Examples |
|---|---:|
| train_known | 31,360 |
| validation_known | 3,920 |
| test_known | 3,920 |
| calibration_unknown | 5,600 |
| test_unknown | 11,200 |

Relation allocation:

- 56 known relations;
- 8 separate relations used only for unknown-threshold calibration;
- 16 untouched relations used as the final unseen-relation test.

## Frozen V1 evaluation

### Known-relation test

| Metric | Result |
|---|---:|
| Accuracy | **89.21%** |
| Macro-F1 | **89.16%** |
| Micro-F1 | **89.21%** |
| Top-3 accuracy | **98.01%** |
| ECE | **1.95%** |

### Final unseen-relation test — balanced profile

| Metric | Result |
|---|---:|
| OOD AUROC | **80.33%** |
| OOD AUPRC | **90.50%** |
| FPR95 | **57.86%** |
| Unknown-F1 | **85.30%** |
| Known acceptance | **61.25%** |
| Unknown rejection | **84.45%** |

The balanced threshold `0.967` was frozen from validation-known + calibration-unknown data before the final unseen test.

## Operating profiles

| Profile | Threshold | Known coverage | Accuracy on accepted known | Unknown rejection | Unknown-F1 |
|---|---:|---:|---:|---:|---:|
| High coverage | 0.924 | 72.40% | 97.67% | 73.79% | 80.44% |
| **Balanced** | **0.967** | **61.25%** | **98.67%** | **84.45%** | **85.30%** |
| High safety | 0.978 | 55.43% | 98.94% | 88.63% | 86.80% |

`balanced` remains the production default.

## Selective prediction

Known-test risk decreases as the model abstains on lower-confidence examples:

| Coverage | Error risk | Accuracy |
|---:|---:|---:|
| 100% | 10.8% | 89.2% |
| 90% | 6.2% | 93.8% |
| 80% | 3.4% | 96.6% |
| 70% | 1.9% | 98.1% |
| 60% | 1.3% | 98.7% |
| 50% | 0.9% | 99.1% |

This curve concerns selective classification among known examples only.

## Entity-information ablation

ModernBERT-base was evaluated under matched conditions on the Closed-Set-80 protocol:

| Condition | Input | Pooling | Test Accuracy | Test Macro-F1 | Top-3 |
|---|---|---|---:|---:|---:|
| A | raw | CLS | 63.98% | 63.02% | 86.73% |
| B | marked | CLS | 79.18% | 78.84% | 93.13% |
| C | marked | entity-pair | **85.02%** | **84.89%** | **95.96%** |

Test Macro-F1 effects:

- entity markers: **+15.82 percentage points**;
- entity-aware pooling beyond markers: **+6.05 percentage points**;
- full structured entity information vs raw CLS: **+21.87 percentage points**.

## Latency

### GPU benchmark

Sequential open-set inference:

- mean: **22.04 ms**
- median: **21.81 ms**
- P95: **23.32 ms**
- P99: **28.49 ms**
- throughput: **45.37 examples/s**

Batch-8 reached approximately **191 examples/s**.

### Cloud Run CPU — warm V1 benchmark

Production configuration: **2 vCPU / 4 GiB**.

- application median: **87.2 ms**
- application mean: **94.2 ms**
- end-to-end client median: **138.0 ms**
- end-to-end client mean: **157.3 ms**

These figures measure primary inference only. V2 word-occlusion explanation intentionally runs asynchronously in the UI because perturbation analysis performs additional model evaluations.

## V2 system architecture

```text
Browser
   |
   v
PUBLIC Next.js Cloud Run
   |
   | Google ID token
   v
PRIVATE FastAPI Cloud Run
   |
   +--> ModernBERT inference
   +--> word influence
   +--> telemetry
   +--> feedback
   +--> model registry
   |
   +------> GCS model bucket
   |
   +------> GCS operations bucket
   |
   +------> Cloud Tasks
                |
                | OIDC
                v
          private batch callback
```

## API

V1 remains backward compatible:

```text
GET  /v1/model
POST /v1/relations:predict
POST /v1/relations:predict-batch
POST /v1/relations:predict-open-set
POST /v1/relations:explain
```

V2:

```text
GET  /v2/model
GET  /v2/relations
GET  /v2/models
GET  /v2/metrics/summary

POST /v2/relations:predict-open-set
POST /v2/relations:explain
POST /v2/feedback
POST /v2/batch-jobs
GET  /v2/batch-jobs/{job_id}
```

The internal Cloud Tasks callback exists under `/v2/internal/` but is not exposed by the public Next.js proxy.

See [`docs/api.md`](docs/api.md).

## Frontend

The V2 Next.js interface contains:

- Extract;
- Async Batch;
- Ops;
- Model Lab.

Extract includes open-set operating profiles, top-k relations, decision telemetry, word influence, and explicit human feedback.

Ops exposes privacy-preserving production metrics and the current model-registry state.

## Model lifecycle commands

Register:

```bash
uv run python scripts/register_model_release.py \
  artifacts/open_set/best \
  --bucket YOUR_MODEL_BUCKET \
  --model-version modernbert-open-set-v1.1
```

Promote:

```bash
uv run python scripts/promote_model_release.py \
  --bucket YOUR_MODEL_BUCKET \
  --model-version modernbert-open-set-v1.1
```

Export consented feedback:

```bash
uv run python scripts/export_feedback_dataset.py \
  artifacts/review/feedback.jsonl \
  --ops-bucket YOUR_OPS_BUCKET
```

## Repository layout

```text
.
├── apps/web/                     # Next.js application
├── docs/                         # Methodology, architecture, V2 and deployment docs
├── infra/terraform/              # GCP infrastructure as code
├── reports/                      # Frozen evaluation reports
├── scripts/                      # Reproduction, training, lifecycle and export CLIs
├── src/relationscope/
│   ├── api/
│   ├── data/
│   ├── document/                 # V3 entity discovery + document graph pipeline
│   ├── inference/
│   ├── legacy/
│   ├── models/
│   ├── ops/                      # telemetry, feedback, jobs, active learning, drift
│   └── training/                 # calibration, dataset snapshots, promotion gates
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

Without Cloud Tasks environment variables, V2 batch jobs use the same durable local JSON format and execute inline for development.

## Deployment

Infrastructure is Terraform-managed.

V2 adds:

- private operations bucket;
- Cloud Tasks queue;
- dedicated task-caller identity;
- task/API IAM;
- V2 runtime environment configuration.

See:

- [`docs/v2.md`](docs/v2.md)
- [`docs/v3.md`](docs/v3.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/gcp_deployment.md`](docs/gcp_deployment.md)

## Methodological guardrails

- Closed-Set-80 is not presented as the official FewRel few-shot leaderboard protocol.
- Final unseen relations are never used for threshold selection.
- Calibration and final-test metrics are reported separately.
- AUPRC is not compared across splits without considering class prevalence.
- GPU and Cloud Run CPU latency are labeled separately.
- Word occlusion is presented as perturbation evidence, not as a causal explanation.
- Feedback is not automatically treated as ground truth.
- Production serving does not trigger model training.
- Raw text is absent from telemetry and requires explicit consent in feedback/review paths.
- Automatic entity discovery is reported separately from relation-classification quality; the heuristic fallback is not presented as a benchmarked NER model.
- Active-learning scores prioritize review but are not labels.
- A challenger cannot be promoted solely by production traffic; frozen offline gates and explicit promotion remain required.

## Data and attribution

FewRel is provided by THUNLP. If using the dataset, cite:

> Han et al. (2018), *FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation*, EMNLP 2018.

Official repository: https://github.com/thunlp/FewRel

Raw datasets, trained weights, Terraform variable files, and credentials are not committed to Git.
