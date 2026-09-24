# API contract

RelationScope exposes model behavior without requiring the browser to know the model implementation.

## Health

`GET /health`

Returns successfully without loading the model. Use it for container liveness.

`GET /ready`

Resolves/downloads the configured model bundle and loads it. Returns HTTP 503 while no valid model is available.

## V1 compatibility API

The V1 routes remain available for clients built against RelationScope v1.0.0:

```text
GET  /v1/model
POST /v1/relations:predict
POST /v1/relations:predict-batch
POST /v1/relations:predict-open-set
POST /v1/relations:explain
```

V1 explanation uses token/subword occlusion and is kept for compatibility.

## V2 model metadata

`GET /v2/model`

Returns:

- promoted model ID;
- device;
- number of known relations;
- maximum input length;
- pooling mode;
- raw vs marked input view;
- whether open-set calibration is available;
- default and available operating profiles.

## V2 relation catalog

`GET /v2/relations`

Returns the known relation IDs and human-readable names from the loaded model bundle. The frontend uses this catalog when a reviewer corrects a prediction.

## V2 open-set prediction

`POST /v2/relations:predict-open-set`

Example body:

```json
{
  "text": "Marie Curie was born in Warsaw in 1867.",
  "head": {
    "text": "Marie Curie",
    "start": 0,
    "end": 11
  },
  "tail": {
    "text": "Warsaw",
    "start": 24,
    "end": 30
  },
  "top_k": 5,
  "profile": "balanced"
}
```

Offsets are zero-based and `end` is exclusive.

The response contains calibrated top-k relation probabilities plus:

- `model_id`;
- `KNOWN_RELATION` or `UNKNOWN_RELATION`;
- selected operating profile;
- frozen threshold;
- learned temperature;
- inference latency.

V2 prediction metadata is also recorded in the privacy-preserving telemetry store. Raw text is not written to telemetry.

## V2 word influence explanation

`POST /v2/relations:explain`

Example:

```json
{
  "text": "Marie Curie was born in Warsaw in 1867.",
  "head": {
    "text": "Marie Curie",
    "start": 0,
    "end": 11
  },
  "tail": {
    "text": "Warsaw",
    "start": 24,
    "end": 30
  },
  "profile": "balanced",
  "target_relation_id": "P19"
}
```

The endpoint groups tokenizer pieces into complete displayed words, masks one word at a time, evaluates all perturbed examples in a batch, and reports the change in calibrated confidence for the target relation.

For each word:

- positive `confidence_drop`: supports the target relation;
- negative `confidence_drop`: opposes the target relation;
- `normalized_influence`: signed value in [-1, 1] for UI intensity.

The result is perturbation-based evidence. It must not be interpreted as a causal explanation.

## V2 feedback

`POST /v2/feedback`

Supported verdicts:

- `confirm`;
- `correct`;
- `unknown`;
- `skip`.

A correction must contain `corrected_relation_name`.

Privacy behavior:

- raw sentence/entity values are not stored by default;
- hashes and structured prediction metadata are stored;
- raw sentence + entity spans are persisted only when `consent_to_store_text=true`.

The frontend obtains valid known labels from `GET /v2/relations`.

## V2 operational metrics

`GET /v2/metrics/summary?limit=500`

Returns a bounded recent-event aggregate containing:

- prediction count;
- KNOWN / UNKNOWN rates;
- mean confidence;
- mean latency;
- p50 latency;
- p95 latency;
- counts by profile;
- counts by model version.

This is portfolio-scale operational telemetry, not a replacement for a high-throughput observability platform.

## V2 model registry

`GET /v2/models`

Returns immutable model releases discovered under `models/releases/` plus the currently promoted model.

Registration and promotion are explicit CLI actions; the serving API does not train or promote models.

## V2 asynchronous batch jobs

Create:

`POST /v2/batch-jobs`

The API accepts up to 2,000 entity-pair examples per durable job.

Read state/results:

`GET /v2/batch-jobs/{job_id}`

Statuses:

```text
queued -> running -> succeeded
                  -> failed
```

Production jobs are dispatched through Cloud Tasks and run in vectorized chunks.

Internal callback:

`POST /v2/internal/batch-jobs/{job_id}:run`

This route is for the dedicated Cloud Tasks service identity only. The public Next.js proxy blocks `/v2/internal/*`.

## Logging policy

The HTTP middleware logs only request metadata:

- request ID;
- method;
- path;
- response status;
- latency.

Prediction telemetry stores structured model metadata and a SHA-256 digest of the input, not raw sentences.

Human-review text can be stored only through the explicit feedback consent flag.


## V3 document intelligence

### Analyze raw text

`POST /v3/documents:analyze`

The caller provides document text, not entity spans. The response contains:

- automatic entities with canonical IDs and mention offsets;
- sentence-level candidate-pair counts;
- accepted and optionally unknown relation edges;
- document graph metadata;
- active-learning candidate/queue counts;
- relation and NER model IDs;
- end-to-end document-analysis latency.

### Upload PDF/TXT

`POST /v3/documents:upload`

Multipart form fields:

- `file`: PDF or UTF-8 text;
- `profile`: open-set operating profile;
- `include_unknown`: whether UNKNOWN edges appear in the graph;
- `max_candidate_pairs`: bounded pair-generation cap;
- `consent_to_store_review_data`: explicit opt-in for uncertain raw sentence pairs entering the private review queue.

Uploads are limited to 10 MiB.

## V3 continuous learning

Public aggregate endpoints:

```text
GET /v3/active-learning/policy
GET /v3/drift
GET /v3/canary
GET /v3/learning/summary
```

Private reviewer/admin endpoints:

```text
GET  /v3/internal/review-queue
POST /v3/internal/review-queue/{review_id}:review
POST /v3/internal/challengers
```

The public Next.js proxy blocks every `/v3/internal/*` route.

### Drift

`GET /v3/drift` compares recent production telemetry with the preceding window using UNKNOWN-rate, confidence, input-length and predicted-relation-distribution signals.

### Learning summary

`GET /v3/learning/summary` exposes aggregate counts and canary/drift state only. It never returns review text.

### Review queue

The review queue stores raw text only for callers that explicitly consented during V3 document analysis.

Reviewer decisions are:

- `confirm`;
- `correct`;
- `unknown`;
- `dismiss`.

Reviewed examples may later enter a versioned dataset snapshot.

## V3 deployment variant header

The Next.js proxy adds:

```text
x-relationscope-variant: champion
```

or:

```text
x-relationscope-variant: challenger
```

for proxied responses when canary routing is configured.
