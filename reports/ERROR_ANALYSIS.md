# Out-of-fold error analysis

## Scope

This report analyzes one group-isolated prediction for each of the 71,117 usable
records. It uses the already selected entity-aware architecture with `C=0.5`; it is
a post-selection diagnostic, not a new independent benchmark. Machine-readable
evidence, including 25 high-confidence representative errors, is in
[`error_analysis.json`](error_analysis.json).

The diagnostic macro-F1 is 0.5686, consistent with the nested benchmark mean. Each
error example represents a distinct connected tweet-ID/exact/normalized-text family,
so augmented variants cannot dominate the review queue.

## What fails

### Classes

- Negative F1: 0.6648
- Positive F1: 0.5869
- Neutral F1: 0.5540
- Irrelevant F1: 0.4687

The largest directional confusions are Positive → Neutral (3,203 rows), Neutral →
Positive (2,948), Neutral → Negative (2,750), and Positive → Irrelevant (2,650).
The target mixes two decisions—entity relevance and polarity—which a single flat
four-class classifier must learn simultaneously.

### Text length

- 1–40 characters: 0.5092 macro-F1
- 41–80: 0.5611
- 81–160: 0.6033
- 161+: 0.5569

Very short messages provide too little context, while the longest messages lose some
of the mid-length advantage. Confidence should therefore not be interpreted as
calibrated certainty, especially for short inputs.

### Entities

Among entities with at least 500 records, the lowest macro-F1 slices are PUBG
(0.3848), AssassinsCreed (0.4047), CallOfDutyBlackopsColdWar (0.4169), MaddenNFL
(0.4242), and FIFA (0.4273). MaddenNFL illustrates why accuracy alone is misleading:
its accuracy is 0.7814 while its macro-F1 is only 0.4242, indicating poor minority-
class behavior within that entity.

Entity-frequency quartiles are not monotonic. The low-frequency band is weakest at
0.5122 macro-F1, but the least-frequent band reaches 0.5376 and the high-frequency
band 0.5972. Frequency alone does not explain entity difficulty; label mix and
entity-specific language also matter.

## Data-quality findings

Normalization found 1,540 meaningful text families with multiple raw surface forms,
covering 3,410 rows; the largest contains 25 rows. These families are now connected
before splitting, so none crosses an evaluation boundary.

Manual review of representative, high-confidence errors shows both genuine model
failures and plausibly ambiguous labels. Strongly negative wording can be labeled
Neutral, while positive wording about a different game is correctly labeled
Irrelevant by the dataset but attracts a Positive prediction. This is evidence for a
task-structure problem, not merely a need for a larger text encoder.

## Prioritized next experiments

1. Train a hierarchical model: target-entity relevance first, then polarity for
   relevant text. Compare it on the identical connected outer folds.
2. Audit a stratified sample of high-confidence errors for label noise and publish
   an annotation rubric plus inter-annotator agreement.
3. Add calibrated probabilities and an abstention threshold, evaluated with
   reliability curves and expected calibration error.
4. Benchmark a compact transformer on the same grouped folds, prioritizing the
   short-text and worst-entity slices rather than aggregate F1 alone.
