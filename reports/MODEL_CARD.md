# Model card

## Model overview

The model is a classical multiclass NLP baseline that combines word-level and character-level TF-IDF features with class-balanced logistic regression. Preprocessing and classification are persisted as one scikit-learn pipeline so inference cannot accidentally use a different vectorizer.

The design favors interpretability, fast CPU inference, deterministic experimentation, and a strong engineering baseline over architecture novelty.

## Intended use

The model demonstrates leakage-aware sentiment classification for entity-focused social-media text. It is appropriate for portfolio review, education, offline experimentation, and as a baseline for future research.

It is not designed for production moderation or decisions about people. Do not use predictions to infer intent, identity, mental health, personal traits, or eligibility for services.

## Training and evaluation protocol

After schema validation, records with unusable text and exact duplicate rows are removed. Connected components link records that share either a tweet ID or exact text. A shuffled five-fold `StratifiedGroupKFold` with seed 42 provides one untouched holdout fold. The remaining data uses three-fold grouped cross-validation to choose logistic-regression `C` from `0.5`, `1.0`, and `2.0` by macro-F1.

All learned feature extraction occurs inside the cross-validated pipeline. The holdout does not participate in vectorizer fitting, model fitting, or hyperparameter selection.

## Verified performance

Run date: 2026-10-20. Environment: Python 3.12.0, pandas 2.3.3, scikit-learn 1.9.1.

- Development rows: 56,907; untouched holdout rows: 14,210.
- Shared tweet IDs across the boundary: 0.
- Shared exact texts across the boundary: 0.
- Selected regularization: `C=0.5`.
- Grouped-CV macro-F1: 0.5313.
- Holdout accuracy: 0.5559.
- Holdout macro precision: 0.5395.
- Holdout macro recall: 0.5401.
- Holdout macro-F1: 0.5396.
- Per-class F1: Negative 0.6538; Positive 0.5883; Neutral 0.5034; Irrelevant 0.4130.

These results should not be compared directly with the legacy notebook's 84.9% score because that experiment used a leaky row-random split and test-set model selection.

## Limitations

The weakest class is `Irrelevant`, and overall performance is insufficient for high-impact use. The classifier is trained on historical English social-media data about 32 entities and may fail on new entities, dialects, code-switching, sarcasm, reclaimed slurs, emerging vocabulary, or text that requires conversational context. Grouping blocks known exact/ID duplicates but cannot guarantee removal of every semantic near-duplicate.

The model may reproduce dataset biases. Aggregate metrics can hide disparities for linguistic communities that are not identified in the data, so subgroup fairness cannot be established from this dataset.

## Reproducibility and artifacts

The full evidence is stored in [`experiment.json`](experiment.json), including the input hash, cleaning counts, class balance, split diagnostics, confusion matrix, selected parameters, and environment versions. Run `sentiment-train` to reproduce the protocol. Floating-point results can vary slightly across platforms and future compatible dependency releases.

The generated `joblib` artifact is intentionally excluded from Git because it is reproducible and binary. Joblib deserialization can execute arbitrary code; load only trusted artifacts.
