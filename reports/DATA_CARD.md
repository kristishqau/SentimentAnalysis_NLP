# Data card

## Summary

This repository uses the four-column files commonly distributed as the [Twitter Entity Sentiment Analysis dataset on Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). Each headerless record contains a tweet ID, target entity, one of four sentiment labels, and text.

The repository includes `twitter_training.csv` and `twitter_validation.csv` for reproducibility. The project trains only from the training file. The supplied validation file is unsuitable as an independent holdout because every one of its 1,000 IDs is also present in the training file and 516 texts are exact cross-file matches.

## Schema

- `tweet_id`: identifier shared by families of augmented text variants.
- `entity`: one of 32 products, games, companies, or topics.
- `sentiment`: `Negative`, `Neutral`, `Positive`, or `Irrelevant`.
- `text`: raw social-media text; it may contain URLs, handles, emoji, profanity, or encoding artifacts.

## Audited training-file profile

- 74,682 raw records.
- 71,117 usable, unique records after deterministic cleaning.
- 12,447 unique tweet IDs and 32 entities.
- 858 records removed for absent or empty text.
- 2,707 exact duplicate usable rows removed.
- Usable labels: 21,571 Negative; 19,557 Positive; 17,541 Neutral; 12,448 Irrelevant.
- SHA-256: `68ebaee4f44bda5de49b96b5e5630184f1af894970b994aa6bb0aa5e1dbfe9a0`.

## Quality and leakage risks

Tweet IDs identify clusters of near-duplicate augmented examples, so a row-random split overstates generalization. Exact text and surface variants can also appear under different IDs. The package builds connected components over tweet ID, exact-text hash, and conservatively normalized text, then assigns each component wholly to one fold. Every verified fold has zero ID, exact-text, and normalized-text overlap.

Other known limitations include mislabeled or ambiguous sentiment, synthetic-looking text perturbations, mojibake, class imbalance, entity-specific vocabulary, and historical domain shift. Exact grouping does not prove that all semantic near-duplicates have been discovered.

## Responsible use

The data contains profanity, harassment, violent language, and other potentially offensive material. It should not be presented to users without appropriate content controls. Labels describe sentiment toward an entity, not a person's stable emotional state or attributes.

Suitable uses include education, reproducible experimentation, and comparison of leakage-aware text-classification methods. It is not suitable for automated moderation, employment, healthcare, policing, credit, or other consequential decisions.

## Licensing and provenance boundary

The maintained Kaggle mirror identifies the source dataset as `CC0-1.0` / Public
Domain, consistent with independent dataset indexes. This metadata was checked on
2026-10-21. Exact hashes and source links are recorded in the
[`data/` provenance note](../data/README.md).

The repository's MIT license still covers only original code and project
documentation; dataset licensing is a separate grant. CC0 metadata does not remove
privacy, platform, third-party-content, removal-request, or applicable-law concerns.
Users should reassess these obligations before commercial or user-facing use.
