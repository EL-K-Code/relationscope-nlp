# RelationScope architecture

## Design objective

Keep the NLP problem central while making the resulting system deployable, inspectable, privacy-aware and operationally reproducible.

## Research plane

```text
FewRel
  |
  +--> ENSAI lineage
  |     LSTM / Seq2Seq Attention / T5 / scaling
  |
  +--> Closed-Set-80
  |     TF-IDF baseline
  |     Transformer
  |     marker ablation
  |
  +--> Open-set protocol
        calibration
        abstention
        OOD metrics
        selective prediction
```

The V1 benchmark remains frozen. V2 adds system capabilities around the promoted model rather than retuning against the final unseen-relation test.

## Online serving plane

```text
Browser
   |
   v
PUBLIC Next.js Cloud Run
   |
   | server-side Google ID token
   v
PRIVATE FastAPI Cloud Run
   |
   +--> entity validation / marking
   +--> ModernBERT
   +--> entity-pair pooling
   +--> temperature scaling
   +--> operating-profile threshold
   +--> relation / UNKNOWN_RELATION
   |
   +--> word-occlusion influence
   +--> privacy-preserving telemetry
   +--> feedback capture
```

The browser never receives Google service-account credentials.

## Explainability plane

The ENSAI sequence-to-sequence model exposed decoder attention matrices. The production ModernBERT classifier has a different architecture, so V2 uses relation-specific word occlusion rather than presenting self-attention as a direct explanation.

```text
original sentence
      |
      +--> baseline calibrated target confidence
      |
      +--> mask word 1 --> target confidence
      +--> mask word 2 --> target confidence
      +--> ...
      +--> mask word N --> target confidence
                         |
                         v
            signed confidence change
                         |
                         v
              word influence map
```

The perturbed variants are evaluated together as a batch.

## Asynchronous batch plane

```text
Next.js
   |
   v
FastAPI creates durable job
   |
   +--> GCS input/status object
   |
   v
Cloud Tasks queue
   |
   | OIDC token from dedicated task identity
   v
PRIVATE internal worker callback
   |
   v
vectorized ModernBERT chunks
   |
   v
GCS result/status objects
```

The public Next.js proxy rejects `/v2/internal/*`, so the worker callback is not surfaced through the public application.

## Feedback and retraining-data plane

```text
prediction
   |
   v
human verdict
   |
   +--> no consent --> hashes + structured metadata only
   |
   +--> explicit consent --> raw sentence + entity spans
                               |
                               v
                        reviewed export JSONL
                               |
                               v
                       offline retraining only
```

Production feedback never triggers automatic self-training.

## Model lifecycle plane

```text
training output
      |
      v
SHA-256 manifest
      |
      v
models/releases/<version>   <- immutable registered release
      |
      | explicit promotion
      v
models/current              <- serving alias
      |
      v
promotion audit record
```

Model lifecycle scripts live under `scripts/register_model_release.py` and `scripts/promote_model_release.py`.

## Storage plane

Two buckets have distinct permissions and lifecycles:

```text
model bucket
  API: objectViewer
  contains:
    models/current/
    models/releases/
    models/promotions/

operations bucket
  API: objectAdmin
  contains:
    v2/telemetry/
    v2/feedback/
    v2/batch/
```

The operations bucket has a configurable lifecycle retention window, 90 days by default.

## Security boundary

Only the web Cloud Run service is public.

Principals:

- `relationscope-web`: may invoke the API;
- `relationscope-api`: may read models, write operational data, enqueue Cloud Tasks and act as the task caller when creating OIDC tasks;
- `relationscope-tasks`: may invoke the private API worker callback.

## Privacy boundary

- application request logging excludes request bodies and entity text;
- prediction telemetry excludes raw sentence/entity values;
- feedback raw text is opt-in;
- Terraform variable files, model weights and credentials remain outside Git.

## Separation of concerns

Training is not initiated by the serving application.

A model reaches production only through the explicit lifecycle:

```text
train -> evaluate -> promotion gates -> register -> promote -> serve
```
