# Model promotion gates

A model is not promoted to the GCS `models/current` prefix simply because it has the highest accuracy.

## Required evidence

Before promotion, the candidate must have:

- a frozen dataset protocol;
- a committed training script;
- a committed evaluation script;
- closed-set Macro-F1 and ECE;
- error analysis;
- latency benchmark on the intended serving hardware;
- model bundle manifest;
- for open-set models: threshold selected without using the final unseen-relation test set.

## Default comparison gates

These are decision rules, not pre-filled performance claims.

A candidate should normally:

1. beat the TF-IDF baseline on Macro-F1;
2. not materially regress high-frequency relations to gain average accuracy;
3. improve or preserve calibration after temperature scaling;
4. meet the latency budget chosen for the public demo;
5. pass marker-ablation sanity checks;
6. show a useful risk/coverage trade-off if abstention is enabled.

## Promotion record

For every promoted model, record:

```text
git commit
dataset protocol
seed
base model
training config
test metrics
calibration metrics
latency
model artifact SHA-256 manifest
GCS version/prefix
promotion date
```

The public README and CV should only use numbers from a promoted/reproduced model.