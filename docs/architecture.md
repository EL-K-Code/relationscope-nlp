# RelationScope architecture

## Design objective

Keep the NLP problem central while making the resulting system deployable and inspectable.

## Research plane

```text
FewRel
  |
  +--> ENSAI lineage
  |     LSTM / Attention / T5 / scaling
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
```

## Serving plane

```text
user
 |
 v
Next.js
 |
 | same-origin proxy
 v
FastAPI
 |
 +--> validation
 +--> entity marking
 +--> tokenizer
 +--> Transformer
 +--> calibration
 +--> top-k / abstention
 +--> optional occlusion evidence
```

## Artifact plane

```text
training output
    |
    v
model bundle
    |
    +--> encoder/
    +--> tokenizer/
    +--> classifier.pt
    +--> relations.json
    +--> model_config.json
    +--> open_set_config.json (optional)
    +--> manifest.json
              |
              v
        versioned GCS
```

## Security boundary

Only the web Cloud Run service is public. FastAPI remains IAM-protected. The web service obtains a Google-signed ID token using its attached service identity and is the only application principal granted Cloud Run Invoker on the API.

## Privacy boundary

Raw request bodies are never written to structured logs by application middleware.

## Separation of concerns

Training is never initiated by the serving application. Deployments promote immutable source/container versions and a separately versioned model bundle.
