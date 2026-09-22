# ENSAI notebook lineage

RelationScope is **not** a replacement for the ENSAI Advanced NLP lab. It is the engineering continuation of it.

## Original notebook

The lab by Maxime Ferret and Alex Komla Labou studied relation extraction as two different formulations:

1. **fixed-label classification**;
2. **sequence-to-sequence generation** of relation descriptions.

The notebook also studied attention and the role of explicit entity markers.

## Original data choices preserved

The notebook combined the 64 FewRel training relations and 16 FewRel validation relations into a 56,000-example pool for its educational closed-set experiments.

It inserted:

- `[HEAD] ... [/HEAD]`
- `[TAIL] ... [/TAIL]`

around the entity spans.

With random seed 42, the initial seven relations were:

| Property | Relation |
|---|---|
| P137 | operator |
| P105 | taxon rank |
| P26 | spouse |
| P2094 | competition class |
| P1877 | after a work by |
| P1411 | nominated for |
| P136 | genre |

The original split was:

```text
all selected examples
      |
      +-- 70% train pool
      |      |
      |      +-- 80% final train
      |      +-- 20% validation
      |
      +-- 30% test
```

For seven FewRel relations this produced:

- 2,744 training examples;
- 686 validation examples;
- 1,470 test examples.

The legacy reproduction scripts intentionally preserve this split.

## Original recurrent baseline

The notebook baseline was:

```text
Embedding(vocab_size, 64, mask_zero=True)
        ↓
LSTM(128)
        ↓
Dense(K, softmax)
```

trained for 10 epochs with Adam and sparse categorical cross-entropy.

The notebook's input vocabulary was built from training data only, lowercased, and capped at 5,000 tokens.

## Original seq2seq experiment

The notebook reframed relation extraction as generation:

```text
marked sentence
     ↓
LSTM encoder
     ↓
Attention
     ↓
LSTM decoder
     ↓
relation description
```

For the seven-relation experiment, the notebook reported a maximum input sequence length of 40 tokens and maximum generated label sequence length of 24 tokens.

Attention scores were visualized against input tokens to inspect whether entity markers guided alignment.

## Original T5 experiment

The notebook fine-tuned `t5-small` with inputs formatted as:

```text
extract relation: <marked sentence>
```

and generated the full human-readable relation name + description.

A later version used beam search.

Important correction in RelationScope:

The notebook used `max_target_length=32`. Several complete relation descriptions exceed this budget, making exact-match evaluation partly a target-truncation test.

RelationScope preserves a **legacy-32 mode** for reproduction, but the corrected experiment first measures target token lengths and chooses a non-truncating target budget.

## Scaling study preserved

The notebook compared:

- baseline LSTM;
- BiLSTM + SpatialDropout + LSTM dropout + final dropout;

for:

```text
K = 7, 10, 15, 20, 30 relations
```

The BiLSTM architecture used:

- embedding dimension 64;
- bidirectional LSTM hidden dimension 128;
- SpatialDropout1D 0.2;
- LSTM dropout 0.2;
- GlobalMaxPooling1D;
- final dropout 0.3.

## RelationScope extension

The production track adds:

- explicit reproducible benchmark names;
- stronger lexical baseline;
- entity-aware Transformer classifier;
- calibration metrics;
- top-k inference;
- batch API;
- Docker/CI;
- future GCP deployment.

The research story is therefore:

```text
LSTM classification
   ↓
Seq2Seq + attention
   ↓
T5 generation
   ↓
scaling / regularization study
   ↓
entity-aware ModernBERT
   ↓
production relation extraction service
```

This lineage should remain visible in the README and future portfolio write-up.
