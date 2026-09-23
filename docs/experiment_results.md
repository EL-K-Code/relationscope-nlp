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

## 5. Final open-set benchmark

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

> Official FewRel validation relations are used only as final unseen-relation test data. Operating thresholds are frozen from validation-known + calibration-unknown data before the final unseen test.

### Training history

| Epoch | Train loss | Validation accuracy | Validation Macro-F1 | Top-3 | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.0224480918 | 0.8311224490 | 0.8281654720 | 0.9604591837 | 0.0230505190 |
| 2 | 0.4106610154 | 0.8905612245 | 0.8896936652 | 0.9772959184 | 0.0113807839 |
| 3 | 0.2122534933 | 0.9012755102 | 0.9008318570 | 0.9808673469 | 0.0213488214 |

Temperature scaling:

- Temperature: **1.0868413448**

### Known-relation final test

- Accuracy: **0.8920918367**
- Macro-F1: **0.8915730906**
- Micro-F1: **0.8920918367**
- Top-3 accuracy: **0.9801020408**
- ECE: **0.0194929293**

### Calibration open-set split

Balanced operating point, threshold **0.967**:

- AUROC: **0.8370367392**
- AUPRC: **0.8680865653**
- FPR95: **0.5905612245**
- Unknown-F1: **0.8181209178**
- Known acceptance: **0.6283163265**
- Unknown rejection: **0.8723214286**

### Final unseen-relation test

The threshold remains frozen at **0.967**.

- AUROC: **0.8032652036**
- AUPRC: **0.9049622513**
- FPR95: **0.5785714286**
- Unknown-F1: **0.8529557650**
- Known acceptance: **0.6125000000**
- Unknown rejection: **0.8444642857**

AUPRC differs with class prevalence and is not interpreted as a direct calibration-vs-final improvement. AUROC is the more prevalence-stable comparison.

## 6. Frozen operating profiles

Thresholds were selected only on validation-known + calibration-unknown data.

### Calibration selection

| Profile | Threshold | Known acceptance | Accepted-known accuracy | Unknown rejection | Unknown-F1 |
|---|---:|---:|---:|---:|---:|
| High coverage | 0.924 | 0.7357 | 0.9795 | 0.8014 | 0.8069 |
| Balanced | 0.967 | 0.6283 | 0.9898 | 0.8723 | 0.8181 |
| High safety | 0.978 | 0.5620 | 0.9923 | 0.9027 | 0.8172 |

### Frozen final test

| Profile | Threshold | Known acceptance | Accepted-known accuracy | Unknown rejection | Unknown-F1 |
|---|---:|---:|---:|---:|---:|
| High coverage | 0.924 | 0.7240 | 0.9767 | 0.7379 | 0.8044 |
| **Balanced** | **0.967** | **0.6125** | **0.9867** | **0.8445** | **0.8530** |
| High safety | 0.978 | 0.5543 | 0.9894 | 0.8863 | 0.8680 |

`balanced` is the production default.

## 7. Risk-coverage on known test data

| Coverage | Risk | Accuracy |
|---:|---:|---:|
| 1.00 | 0.108 | 0.892 |
| 0.90 | 0.062 | 0.938 |
| 0.80 | 0.034 | 0.966 |
| 0.70 | 0.019 | 0.981 |
| 0.60 | 0.013 | 0.987 |
| 0.50 | 0.009 | 0.991 |

This measures selective classification among known examples only.

## 8. Latency

### GPU sequential open-set inference

500 examples:

- Mean: **22.04 ms**
- Median: **21.81 ms**
- P95: **23.32 ms**
- P99: **28.49 ms**
- Min: **20.28 ms**
- Max: **33.87 ms**
- Throughput: **45.37 examples/s**

### GPU batch benchmark

| Batch | Mean batch latency | P95 | Mean per example | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 22.56 ms | 24.42 ms | 22.56 ms | 44.33 ex/s |
| 8 | 41.83 ms | 53.01 ms | 5.23 ms | 191.24 ex/s |
| 16 | 84.45 ms | 107.15 ms | 5.28 ms | 189.46 ex/s |
| 32 | 172.24 ms | 209.87 ms | 5.38 ms | 185.79 ex/s |
| 64 | 374.84 ms | 444.47 ms | 5.86 ms | 170.74 ex/s |
| 128 | 721.39 ms | 791.46 ms | 5.64 ms | 177.43 ex/s |

### Google Cloud Run CPU — warm

Configuration: 2 vCPU, 4 GiB.

10 sequential requests after model warm-up:

- Application inference mean: **94.2 ms**
- Application inference median: **87.2 ms**
- Client end-to-end mean: **157.3 ms**
- Client end-to-end median: **138.0 ms**

These CPU results are intentionally reported separately from the GPU benchmark. They do not include a scale-to-zero cold start.

## 9. Promotion status

### Production promoted

**ModernBERT-base + marked entities + entity-pair pooling + temperature scaling + open-set profiles**

Promoted bundle:

`modernbert-open-set-v1.1`

Production default:

`balanced` at threshold **0.967**

Validated production chain:

1. model training and frozen evaluation;
2. temperature scaling and operating-point calibration;
3. GCS model publication with SHA-256 manifest;
4. CPU-only container image;
5. private FastAPI service on Cloud Run;
6. public Next.js service with server-side authenticated proxy;
7. live production inference;
8. warm Cloud Run CPU latency benchmark.

RelationScope is therefore ready for the **v1.0.0** release freeze.
