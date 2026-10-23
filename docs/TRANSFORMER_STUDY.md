# Transformer optimization study protocol

## Objective

This study closes three remaining evidence gaps: comparison with DistilRoBERTa and
DeBERTa-v3-small, bounded high-impact hyperparameter selection, and random-seed
stability. It is designed before observing any new outer-test result.

## Nested selection

Every canonical outer fold is processed independently:

1. The outer training partition is split again with connected duplicate groups.
2. Four encoders are trained with the same baseline configuration: DistilBERT,
   Twitter-RoBERTa, DistilRoBERTa, and DeBERTa-v3-small.
3. The encoder with the highest inner-validation macro-F1 advances.
4. A coordinate search evaluates learning rates `1e-5`, `2e-5`, `3e-5`, maximum
   lengths `96`, `128`, `192`, and weight decay `0.0`, `0.01` for that encoder.
5. The inner-validation winner is evaluated once on the untouched outer fold.

All candidates use balanced cross-entropy, a four-epoch ceiling, macro-F1 checkpoint
selection, and early stopping. The search is deliberately bounded: it covers the
high-impact choices without presenting a costly Cartesian sweep as scientific rigor.
The optional dependency set includes both SentencePiece and protobuf because the
DeBERTa-v3 tokenizer requires them during vocabulary conversion.

## Seed policy

The selected configuration is retrained with seeds 42, 43, and 44. Seed 42 remains
the canonical fold score; all three results quantify stability. A seed is never
chosen because it performs best on the outer test fold. This avoids a subtle form of
test-set tuning.

## Recovery and provenance

Each outer fold can be run and downloaded independently. The merge command accepts a
complete set only when folds 1–5 are unique and their dataset and protocol metadata
match. The existing benchmark reference additionally enforces the canonical dataset
hash, scikit-learn version, and exact outer-fold sizes.

On a free Colab GPU, run:

```bash
sentiment-transformer-study \
  --outer-fold 1 \
  --epochs 4 \
  --batch-size 32 \
  --reference reports/benchmark_v2.json \
  --output artifacts/transformer_study_fold_1.json
```

Repeat for folds 2–5, then merge:

```bash
sentiment-transformer-study-merge \
  artifacts/transformer_study_fold_1.json \
  artifacts/transformer_study_fold_2.json \
  artifacts/transformer_study_fold_3.json \
  artifacts/transformer_study_fold_4.json \
  artifacts/transformer_study_fold_5.json \
  --output reports/transformer_optimization_study.json
```

The Colab notebook automates these commands and downloads every completed fold
immediately. A full study performs 55 fine-tuning runs, so run one outer fold at a
time and retain its JSON before relying on a free-session quota.

## Publication gate

Do not update the headline metric from partial folds. Publication requires all five
fold reports, a successful strict merge, evidence regression tests, and a review of
class-level and runtime trade-offs. Until then, the validated 63.55% Twitter-RoBERTa
macro-F1 remains the project headline.
