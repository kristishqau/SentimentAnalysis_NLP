# Model card

## Model overview

The v2 model is a classical multiclass NLP system that combines word-level and character-level TF-IDF, one-hot entity context, and class-balanced logistic regression. Preprocessing and classification are persisted as one scikit-learn pipeline so inference cannot accidentally use different feature transformations.

The design favors interpretability, fast CPU inference, deterministic experimentation, and a strong engineering baseline over architecture novelty.

## Intended use

The model demonstrates leakage-aware sentiment classification for entity-focused social-media text. It is appropriate for portfolio review, education, offline experimentation, and as a baseline for future research.

It is not designed for production moderation or decisions about people. Do not use predictions to infer intent, identity, mental health, personal traits, or eligibility for services.

## Training and evaluation protocol

After schema validation, records with unusable text and exact duplicate rows are removed. Connected components link records that share a tweet ID, exact text, or conservatively normalized near-duplicate text. V2 compares four architectures with shuffled five-fold outer `StratifiedGroupKFold` using seed 42. Within every non-trivial candidate and outer fold, three-fold grouped cross-validation chooses logistic-regression `C` from `0.5`, `1.0`, and `2.0` by macro-F1.

All learned feature extraction occurs inside the cross-validated pipeline. Each outer test fold is excluded from vectorizer fitting, model fitting, and hyperparameter selection. After evaluation, the winning architecture is tuned and fitted on every usable row to produce the inference artifact.

## Verified performance

Run completed: 2026-10-21. Environment: Python 3.12.0, pandas 2.3.3, scikit-learn 1.9.1.

- Usable rows: 71,117 across 12,447 tweet IDs and 32 entities.
- Shared tweet IDs in every outer fold: 0.
- Shared exact texts in every outer fold: 0.
- Shared normalized near-duplicate texts in every outer fold: 0.
- Selected regularization: `C=0.5`.
- Mean outer-fold accuracy: 0.5811 (standard deviation 0.0095).
- Mean outer-fold macro-F1: 0.5686 (standard deviation 0.0090).
- Mean per-class F1: Negative 0.6648; Positive 0.5867; Neutral 0.5540; Irrelevant 0.4689.
- Improvement over word+character without entity context: 0.0268 macro-F1.

See the [v2 benchmark report](BENCHMARK.md) for every baseline and the runtime caveat. These results should not be compared directly with the legacy notebook's 84.9% score because that experiment used a leaky row-random split and test-set model selection.

## Limitations

The weakest class is `Irrelevant`, and overall performance is insufficient for high-impact use. The classifier is trained on historical English social-media data about 32 entities and may fail on new entities, dialects, code-switching, sarcasm, reclaimed slurs, emerging vocabulary, or text that requires conversational context. Grouping blocks known exact/ID duplicates but cannot guarantee removal of every semantic near-duplicate.

The [out-of-fold error analysis](ERROR_ANALYSIS.md) shows especially weak results for short messages and several entity slices. It also indicates that the flat target combines entity relevance with polarity and that some labels are plausibly ambiguous. Reported probabilities are not calibrated and must not be treated as reliable certainty.

The model may reproduce dataset biases. Aggregate metrics can hide disparities for linguistic communities that are not identified in the data, so subgroup fairness cannot be established from this dataset.

## Reproducibility and artifacts

The primary evidence is stored in [`benchmark_v2.json`](benchmark_v2.json), including the input hash, cleaning counts, every fold's metrics and leakage diagnostics, uncertainty, selected parameters, timings, artifact sizes, and environment versions. Run `sentiment-benchmark` to reproduce it. [`experiment.json`](experiment.json) remains as historical v1 holdout evidence. Floating-point results and timings can vary across platforms and compatible dependency releases.

The generated `joblib` artifact is intentionally excluded from Git because it is reproducible and binary. Joblib deserialization can execute arbitrary code; load only trusted artifacts.

The local demo exposes maximum softmax probability as explicitly **uncalibrated**
confidence. It is a portfolio interface, not a production service, and binds to
localhost by default. Its health endpoint confirms artifact loading but does not imply
fitness for high-impact use.
