# API contract

## Health

`GET /health`

Does not load the model and can be used as a container liveness probe.

## Model information

`GET /v1/model`

Returns the loaded model ID, device, number of supported relations and maximum input length.

## Single relation prediction

`POST /v1/relations:predict`

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
  "top_k": 3
}
```

Offsets are zero-based and `end` is exclusive. RelationScope validates that the supplied offsets match the entity text exactly.

## Batch prediction

`POST /v1/relations:predict-batch`

Up to 128 examples are encoded in one padded Transformer batch.

```json
{
  "examples": [
    {
      "text": "Marie Curie was born in Warsaw in 1867.",
      "head": {"text": "Marie Curie", "start": 0, "end": 11},
      "tail": {"text": "Warsaw", "start": 24, "end": 30}
    },
    {
      "text": "Apple was founded by Steve Jobs.",
      "head": {"text": "Apple", "start": 0, "end": 5},
      "tail": {"text": "Steve Jobs", "start": 21, "end": 31}
    }
  ],
  "top_k": 5
}
```

The response contains top-k predictions for each entity pair plus total batch latency.

## Error behavior

- invalid or overlapping entity spans: HTTP 422;
- missing model bundle on model-dependent endpoints: HTTP 503 or 422 depending on endpoint;
- health endpoint remains available even before a model bundle is mounted.

## Production note

The current API deliberately requires explicit entity spans. Entity recognition is a separate task and will not be silently hidden inside relation classification. A later frontend may help users select entities interactively.
