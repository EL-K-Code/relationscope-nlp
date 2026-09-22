# Premium research extensions

RelationScope keeps the ENSAI notebook as its experimental lineage and adds three production/research questions that make the project substantially stronger.

## 1. Entity-information ablation

Question:

> How much of relation extraction performance comes from the sentence alone, explicit entity markers, and entity-aware pooling?

Controlled conditions:

| Condition | Input | Pooling |
|---|---|---|
| A | raw sentence | CLS only |
| B | HEAD/TAIL-marked sentence | CLS only |
| C | HEAD/TAIL-marked sentence | CLS + HEAD + TAIL + abs(HEAD-TAIL) |

All three conditions must use the same base Transformer, splits, seed, optimizer family and evaluation metrics.

Run:

```bash
uv run python scripts/run_marker_ablation.py
```

Primary metric: Macro-F1. Secondary metrics: accuracy, top-3 accuracy and ECE.

No conclusion about entity markers should be written before this controlled comparison is reproduced.

## 2. Open-set unseen-relation detection

Closed-set classifiers always choose one known relation, even when the true relation has never appeared in training.

RelationScope therefore defines:

- 56 official FewRel train relations = known training classes;
- 8 official FewRel train relations = calibration-unknown classes;
- 16 official FewRel validation relations = final test-unknown classes.

The 16 final unseen relations are never used for model selection or threshold tuning.

Prepare:

```bash
uv run python scripts/prepare_open_set_fewrel.py
```

Train/evaluate:

```bash
uv run python scripts/train_open_set_transformer.py
```

The system fits temperature scaling on known validation logits, tunes an abstention threshold using the separate calibration-unknown relations, then evaluates on the untouched 16-relation unseen test set.

Report:

- known-class Macro-F1;
- ECE after calibration;
- OOD AUROC;
- OOD AUPRC;
- FPR@95 TPR;
- unknown-relation F1;
- known acceptance rate;
- unknown rejection rate;
- risk-coverage curve.

A successful open-set bundle stores the learned temperature and threshold and can return:

```text
UNKNOWN_RELATION
```

instead of forcing a known class.

## 3. Evidence by token occlusion

The original notebook visualized attention.

The production model exposes a deliberately different explanation:

1. predict the relation;
2. mask one token at a time;
3. rerun the model in a single occlusion batch;
4. measure the drop in confidence for the predicted relation.

The API returns tokens ranked by confidence drop.

Endpoint:

```text
POST /v1/relations:explain
```

This is presented as **occlusion evidence**, not as a causal explanation of the model.

## Why these extensions matter

The research progression remains coherent:

```text
LSTM
  ↓
Seq2Seq + Attention
  ↓
T5
  ↓
Scaling study
  ↓
Entity-marker ablation
  ↓
Entity-aware Transformer
  ↓
Calibration / abstention
  ↓
Unseen-relation detection
  ↓
Production API + explanations
```

The project therefore demonstrates both classical NLP experimentation and production-grade model behavior.
