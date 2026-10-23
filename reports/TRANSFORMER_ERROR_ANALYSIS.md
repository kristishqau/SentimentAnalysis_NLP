# Transformer error analysis

## Scope

This diagnostic pools the five group-isolated outer-fold confusion matrices for the
selected `cardiffnlp/twitter-roberta-base` benchmark. It covers all 71,117 usable
records exactly once. It is a post-selection analysis, not a new model comparison.
The machine-readable report records the source evidence SHA-256 in
[`transformer_error_analysis.json`](transformer_error_analysis.json).

Pooling predictions produces 64.85% accuracy and 63.58% macro-F1. The pooled
macro-F1 differs slightly from the published 63.55% mean-of-folds estimate because
the two statistics weight fold-level class rates differently. Model selection and
comparison continue to use the preregistered mean-of-folds metric.

## Class findings

- Negative: 73.36% F1 (73.31% mean fold F1)
- Positive: 66.22% F1 (66.13% mean fold F1)
- Neutral: 59.51% F1 (59.48% mean fold F1)
- Irrelevant: 55.23% F1 (55.25% mean fold F1)

`Irrelevant` remains the weakest class, but its mean fold F1 is 8.36 points above
the classical model's 46.89%. Its fold F1 ranges from 53.29% to 58.45%, showing that
the limitation is consistent rather than caused by one anomalous split.

## Dominant confusions

The three largest directional errors are Neutral -> Positive (3,225 rows), Positive
-> Neutral (2,439), and Neutral -> Negative (2,385). For `Irrelevant`, the largest
error is Irrelevant -> Positive (2,069), followed by Irrelevant -> Negative (1,738)
and Irrelevant -> Neutral (1,651).

These patterns support a two-part interpretation of the task: the model must first
infer whether the message concerns the named entity, then infer polarity. A flat
four-class objective entangles those decisions. Balanced cross-entropy materially
helps the minority class, but does not resolve ambiguous relevance boundaries or
label noise.

## Actionable follow-up

1. Preserve row-level out-of-fold probabilities in the next GPU study so entity,
   length, confidence, and calibration slices can be measured for the transformer.
2. Compare a hierarchical relevance-then-polarity objective only on the same
   canonical outer folds.
3. Audit a stratified sample of high-confidence `Irrelevant` errors without changing
   evaluation labels, and publish an annotation rubric before any relabeling study.
