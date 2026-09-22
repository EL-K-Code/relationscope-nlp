# Methodology

## Origin

The original ENSAI lab explored relation extraction through an encoder-only LSTM classification baseline, an LSTM encoder-decoder with attention, T5 generation of textual relation descriptions, and scaling experiments from 7 to 30 relations.

That notebook is valuable experimental history, but its code and reported metrics are not used directly as production claims.

## Corrections introduced in RelationScope

### 1. Protocol is named explicitly

FewRel 1.0 has 64 train relations and 16 disjoint validation relations. Combining both and re-splitting changes the research question.

RelationScope therefore calls its production benchmark **FewRel Closed-Set-80**. It is not a FewRel leaderboard result.

### 2. Split inside each relation

RelationScope performs deterministic per-relation splitting (80/10/10 by default), ensuring every known relation is represented in every closed-set partition.

### 3. Labels are IDs, not generated descriptions

The original T5 experiment generated natural-language relation descriptions. Exact-match scores are sensitive to wording and target truncation.

The primary production task is now classification over stable relation IDs. Human-readable relation names remain metadata.

### 4. Strong simple baseline first

Before training a Transformer, RelationScope measures a word+character TF-IDF logistic-regression baseline.

A Transformer must beat this baseline on Macro-F1 and justify its latency/cost.

### 5. Entity-aware representation

Inputs contain explicit HEAD/TAIL markers. The Transformer pools CLS, HEAD, TAIL and absolute HEAD/TAIL difference representations.

### 6. Calibration is measured

Every benchmark reports accuracy, macro-F1, micro-F1, top-3 accuracy and Expected Calibration Error (ECE).

Future versions add temperature scaling and abstention curves.

## No unsupported claims

Metrics enter the README or CV only after they are reproduced by committed scripts against a frozen protocol.
