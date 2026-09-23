# RelationScope NLP — Experiment Results Registry

> Status: execution phase in progress.
>
> These results were captured from completed experiment logs produced during the V1 execution phase.
> Model checkpoints from the first Kaggle session were not persisted, but the metrics below are preserved.
> Do not treat a model as production-promoted until the open-set, latency, and final promotion checks are complete.

## 1. ENSAI notebook lineage

### LSTM classifier — 7 relations

Protocol: notebook lineage 7-relation closed set

- Test accuracy: **0.9258503401**
- Test macro-F1: **0.9268352780**
- Test micro-F1: **0.9258503401**
- Validation accuracy: **0.9169096210**
- Validation macro-F1: **0.9176548839**

Historical notebook reference: ~0.934 test accuracy.

### Seq2Seq LSTM + Attention — 7 relations

Protocol: notebook seq2seq attention 7-relation closed set

- Teacher-forced test loss: **0.0283932481**
- Teacher-forced test accuracy: **0.6490647197**
- Autoregressive full-sequence exact match: **0.8925170068**

The teacher-forced token accuracy is not directly comparable to autoregressive exact match.

### T5-small — 7 relations

Protocol: notebook T5 relation generation

- Required target length: **37**
- Train labels over legacy target length 32: **1,558**
- Test full-gold exact match: **0.9795918367**
- Test truncated-gold exact match: **0.9795918367**

Training history:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.3881268292 | 0.0019785550 |
| 2 | 0.0038009550 | 0.0021216500 |
| 3 | 0.0023690742 | 0.0007580048 |

Because max target length was increased to 37, full-gold and truncated-gold exact match are identical in this corrected run.

## 2. Notebook relation-cardinality scaling

Validation accuracies:

| K | LSTM | BiLSTM + Dropout |
|---:|---:|---:|
| 7 | 0.7303 | 0.7507 |
| 10 | 0.8071 | 0.7898 |
| 15 | 0.6871 | 0.7265 |
| 20 | 0.6847 | 0.7617 |
| 30 | 0.6371 | 0.7105 |

Test accuracies:

| K | LSTM | BiLSTM + Dropout |
|---:|---:|---:|
| 7 | 0.7490 | 0.7721 |
| 10 | 0.7919 | 0.7838 |
| 15 | 0.6861 | 0.7125 |
| 20 | 0.6802 | 0.7467 |
| 30 | 0.6372 | 0.6980 |

## 3. RelationScope Closed-Set-80 benchmark

Dataset protocol:

- 80 relations
- Train: 44,800
- Validation: 5,600
- Test: 5,600
- Seed: 42

### TF-IDF word + char + SGD logistic baseline

Validation:

- Accuracy: **0.6762500000**
- Macro-F1: **0.6665895772**
- Micro-F1: **0.6762500000**
- Top-3 accuracy: **0.8608928571**
- ECE-15: **0.2109889816**

Test:

- Accuracy: **0.6782142857**
- Macro-F1: **0.6694020837**
- Micro-F1: **0.6782142857**
- Top-3 accuracy: **0.8632142857**
- ECE-15: **0.2137175168**

### ModernBERT-base — marked input + entity-pair pooling

Configuration:

- Base model: `answerdotai/ModernBERT-base`
- Pooling: `entity_pair`
- Input view: `marked`
- Epochs: 3
- Batch size: 16
- Seed: 42

Validation history:

| Epoch | Train loss | Accuracy | Macro-F1 | Top-3 | ECE-15 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.2620018278 | 0.7673214286 | 0.7604518254 | 0.9248214286 | 0.0255209734 |
| 2 | 0.5966127427 | 0.8403571429 | 0.8390642640 | 0.9576785714 | 0.0176901369 |
| 3 | 0.3163067232 | 0.8564285714 | 0.8551980350 | 0.9637500000 | 0.0227146852 |

Final test:

- Accuracy: **0.8501785714**
- Macro-F1: **0.8488793106**
- Micro-F1: **0.8501785714**
- Top-3 accuracy: **0.9596428571**
- ECE-15: **0.0269772060**

Closed-set improvement over TF-IDF baseline:

- Accuracy: **+17.20 percentage points**
- Macro-F1: **+17.95 percentage points**
- Top-3 accuracy: **+9.64 percentage points**
- ECE-15: improved from **21.37%** to **2.70%**

## 4. Entity-information ablation

All conditions use ModernBERT-base and the same Closed-Set-80 protocol.

| Condition | Input | Pooling | Test Accuracy | Test Macro-F1 | Top-3 | ECE-15 |
|---|---|---|---:|---:|---:|---:|
| A | raw | CLS | 0.6398214286 | 0.6301699023 | 0.8673214286 | 0.0820115382 |
| B | marked | CLS | 0.7917857143 | 0.7883999544 | 0.9312500000 | 0.0194093651 |
| C | marked | entity_pair | **0.8501785714** | **0.8488793106** | **0.9596428571** | 0.0269772060 |

Ablation effects on test Macro-F1:

- Entity markers (A → B): **+15.82 percentage points**
- Entity-aware pooling (B → C): **+6.05 percentage points**
- Full structured entity information (A → C): **+21.87 percentage points**

Interpretation: the gain is not explained by the pretrained backbone alone. Explicit entity markers provide the largest improvement, and entity-aware pooling adds a further measurable gain.

## 5. Open-set protocol prepared

Protocol:

`fewrel_open_set_56_known_8_calibration_unknown_16_test_unknown`

Relation counts:

- Known: 56
- Calibration unknown: 8
- Final unseen test unknown: 16

Split sizes:

- train_known: 31,360
- validation_known: 3,920
- test_known: 3,920
- calibration_unknown: 5,600
- test_unknown: 11,200

Leakage guardrail:

> Official FewRel validation relations are used only as final unseen-relation test data.

Open-set training has **not yet been completed**.

## 6. Promotion status

### Current best closed-set candidate

**ModernBERT-base + marked entities + entity-pair pooling**

Current test metrics:

- Macro-F1: **84.89%**
- Accuracy: **85.02%**
- Top-3 accuracy: **95.96%**
- ECE-15: **2.70%**

### Not yet production-promoted

Final production promotion must wait for:

1. Open-set training and unseen-relation evaluation.
2. Temperature scaling / abstention threshold validation.
3. Error analysis.
4. Latency benchmark.
5. Final model bundle publication.
6. API/UI integration smoke test.
7. GCP deployment validation.

The final promoted model may therefore differ from the current best closed-set checkpoint.
