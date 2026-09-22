# API contract

RelationScope exposes model behavior without requiring the browser to know the model implementation.

## Liveness

`GET /health`

Returns successfully without loading the model. Use it for container liveness.

## Readiness

`GET /ready`

Resolves/downloads the configured model bundle and loads it. Returns HTTP 503 while no valid model is available.

## Model metadata

`GET /v1/model`

Returns:

- model ID;
- device;
- number of known relations;
- maximum input length;
- pooling mode;
- raw vs marked input view;
- whether open-set calibration is available.

## Single relation prediction

`POST /v1/relations:predict`

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
  "top_k": 5
}
```

Offsets are zero-based and `end` is exclusive. The API rejects offsets that do not match the supplied entity text.

## Batch prediction

`POST /v1/relations:predict-batch`

Up to 128 entity-pair examples are padded and evaluated in one Transformer batch.

The Next.js frontend can import the following CSV schema and convert it to this endpoint:

```text
id,text,head_text,head_start,head_end,tail_text,tail_start,tail_end
```

## Open-set prediction

`POST /v1/relations:predict-open-set`

Requires a bundle containing `open_set_config.json`.

The response includes:

- calibrated top-k probabilities;
- acceptance threshold;
- learned temperature;
- `KNOWN_RELATION` or `UNKNOWN_RELATION`.

The threshold is trained on calibration-unknown relations, not on the final unseen-relation test split.

## Explanation

`POST /v1/relations:explain`

Uses token occlusion:

1. predict the top relation;
2. mask candidate tokens one at a time;
3. evaluate all occluded variants in a batch;
4. rank tokens by confidence drop.

Returned evidence is explicitly presented as an occlusion signal, not a causal model explanation.

## Logging policy

The request middleware logs only metadata:

- request ID;
- method;
- path;
- response status;
- latency.

It does **not** log sentences, entity strings or request bodies.
