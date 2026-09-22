# Notebook reference metrics

These numbers are copied from the executed ENSAI notebook and serve only as historical reproduction targets.

They are **not** production benchmark claims.

## Seven-relation LSTM

Protocol:
- seven relations selected with seed 42;
- 2,744 train / 686 validation / 1,470 test;
- Embedding(64) → LSTM(128) → Dense(7);
- 10 epochs.

Notebook output:
- test loss: **0.28345**;
- test accuracy: **0.93401**;
- classification-report macro F1: **~0.93**.

Per-class F1 in the notebook:
- P137 operator: 0.93;
- P105 taxon rank: 0.98;
- P26 spouse: 0.93;
- P2094 competition class: 0.98;
- P1877 after a work by: 0.89;
- P1411 nominated for: 0.99;
- P136 genre: 0.84.

## Seq2Seq LSTM + attention

Protocol:
- same seven-relation data;
- max input length: 40;
- max output length: 24;
- teacher forcing;
- 15 epochs.

Notebook output:
- teacher-forced test loss: **0.0276**;
- teacher-forced token-level test accuracy: **0.6485**.

Important:
Teacher-forced token accuracy is not equivalent to full autoregressive relation exact match. RelationScope reports both separately.

## T5 generation

Notebook first full-target exact-match evaluation:
- **0.4102 = 603 / 1,470**.

A later Seq2SeqTrainer evaluation reported:
- validation accuracy: **0.91545**;
- test accuracy: **0.92857**.

However, that experiment used `max_target_length=32` and decoded the **truncated label tensor** as the evaluation target. Several generated examples visibly stop mid-description at the same length.

Therefore RelationScope treats:
- **full-gold exact match** as the proper end-to-end metric;
- **truncated-gold exact match** as a legacy reproduction metric only.

## Scaling study

Validation accuracy reported in the notebook:

| K | Baseline LSTM | BiLSTM + dropout |
|---:|---:|---:|
| 7 | 0.708 | 0.723 |
| 10 | 0.797 | 0.789 |
| 15 | 0.697 | 0.707 |
| 20 | 0.718 | 0.780 |
| 30 | 0.644 | 0.713 |

The notebook's conclusion was that regularization and bidirectionality become more useful as relation cardinality grows.

## Reproduction rule

A value is considered reproduced only when the corresponding committed script regenerates it within an explicitly documented tolerance and environment.

Until then, these remain historical notebook outputs.
