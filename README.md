# SentimentAnalysis_NLP

[![Quality](https://github.com/kristishqau/SentimentAnalysis_NLP/actions/workflows/quality.yml/badge.svg)](https://github.com/kristishqau/SentimentAnalysis_NLP/actions/workflows/quality.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)

A production-minded, classical NLP baseline for four-class, entity-focused tweet sentiment. The project classifies text as **Negative**, **Neutral**, **Positive**, or **Irrelevant** while making data leakage, reproducibility, and artifact provenance first-class concerns.

The central result is not an inflated notebook score: it is a defensible evaluation protocol. Augmented variants in this dataset share tweet IDs, and some exact texts occur under different IDs. This implementation keeps every connected duplicate family inside one fold during holdout creation and model selection.

## Verified result

The checked-in experiment evidence records:

- **55.59% holdout accuracy** and **53.96% macro-F1** on 14,210 untouched rows.
- **53.13% grouped-CV macro-F1** for the selected `C=0.5` model.
- **0 shared tweet IDs** and **0 exact shared texts** across train and holdout.
- A deterministic seed, input SHA-256, cleaning counts, split facts, package versions, per-class metrics, and confusion matrix.

This score is intentionally lower than the original notebook's 84.9% random-split result. That older approach allowed related augmented samples to cross the boundary and selected models against its test set. See the [model card](reports/MODEL_CARD.md) and [machine-readable experiment](reports/experiment.json) for the full evidence.

## Engineering highlights

- End-to-end scikit-learn pipeline: word `(1, 2)` and character `(3, 5)` TF-IDF features plus balanced logistic regression.
- Leakage-safe connected grouping over both tweet ID and exact-text hash.
- Grouped, stratified cross-validation for hyperparameter selection; holdout used once.
- Versioned model artifact containing preprocessing, classifier, and experiment metadata together.
- Installable `src/` package with train and predict CLIs.
- 95% branch coverage, Ruff quality gates, Python 3.10–3.13 CI, dependency auditing, and Dependabot.

```text
headerless CSV
    │
    ├── schema validation + cleaning + SHA-256 provenance
    │
    └── connected duplicate groups (tweet ID ∪ exact text)
            │
            ├── untouched 20% holdout ───────────────┐
            │                                         │ evaluate once
            └── 80% development set                  ▼
                    └── grouped CV → TF-IDF → logistic regression
                                              │
                                              ├── versioned model artifact
                                              └── reproducible metrics JSON
```

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/kristishqau/SentimentAnalysis_NLP.git
cd SentimentAnalysis_NLP
uv sync --locked --all-extras
```

This uses the checked-in `uv.lock` to reproduce the complete development and notebook environment. A conventional pip installation remains supported:

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

Train with the audited defaults:

```bash
uv run sentiment-train
```

The command writes a trusted local pipeline to `artifacts/sentiment_model.joblib` and complete evidence to `artifacts/metrics.json`. Training is CPU-intensive because it performs nine grouped-CV fits.

Predict one or more texts:

```bash
uv run sentiment-predict --model artifacts/sentiment_model.joblib "The update is excellent" "The service is frustrating and broken"
```

`joblib` can execute code during deserialization. Load only artifacts you created or obtained from a trusted source.

## Explore and develop

- [Exploration notebook](notebooks/sentiment_analysis.ipynb) — compact EDA, leakage demonstration, split audit, and result visualization. It calls package code instead of duplicating training logic.
- [Data card](reports/DATA_CARD.md) — provenance, schema, quality findings, responsible-use notes, and licensing boundary.
- [Model card](reports/MODEL_CARD.md) — intended use, methodology, metrics, limitations, and ethical considerations.

Run the same local gates as CI:

```bash
uv run ruff check src tests notebooks
uv run ruff format --check src tests notebooks
uv run pytest
uv run pip-audit
```

## Repository layout

```text
src/sentiment_analysis/   validated data, modeling, artifacts, and CLIs
tests/                    invariant-focused unit and integration tests
notebooks/                recruiter-friendly exploration and audit narrative
reports/                  model card, data card, and experiment evidence
data/                     source CSV files
.github/                  CI and automated dependency maintenance
```

## Scope and limitations

This is an educational portfolio project, not a moderation or decision-making system. The data contains offensive language, platform-specific entities, mislabeled or ambiguous examples, duplicated augmentations, and historical social-media language. Performance—especially for **Irrelevant**—is not adequate for high-impact use. Do not infer emotions, intent, identity, mental state, or personal traits from predictions.

The MIT license covers project code and documentation. Dataset rights are governed separately by the [Kaggle dataset source](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis); review its current terms before redistribution or commercial use.

## License

Code and documentation are available under the [MIT License](LICENSE). Dataset licensing is separate as noted above.
