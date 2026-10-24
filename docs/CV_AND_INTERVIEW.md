# CV and interview guide

## Copy-ready CV entry

**SentimentAnalysis_NLP — leakage-aware NLP evaluation and deployment system**  
Python, scikit-learn, PyTorch, Transformers, pandas, Docker, GitHub Actions

- Engineered a four-class sentiment system as a tested Python package, CLI, browser
  demo, and hardened container for 71,117 entity-focused social-media records.
- Designed connected-group nested evaluation that eliminated tweet-ID, exact-text, and
  normalized-text leakage across all five outer folds.
- Improved macro-F1 from 0.5686 with an entity-aware TF-IDF baseline to 0.6355 with
  Twitter-RoBERTa, an 11.76% relative gain, while publishing uncertainty and
  class-level error analysis.
- Built reproducible evidence with dataset hashes, an exact dependency lock, 30 tests,
  more than 90% coverage, Python 3.11–3.14 CI, dependency auditing, and container
  health and inference checks.

Use three bullets when space is limited: keep bullets one, two, and three.

## 30-second explanation

> I built an auditable entity-focused sentiment system around a dataset containing
> augmented tweet families. I grouped connected duplicates before nested cross-
> validation so related records could not leak across folds. The strongest classical
> pipeline reached 0.5686 macro-F1, and Twitter-RoBERTa reached 0.6355 on the exact
> same five leakage-resistant folds. I packaged training and inference, added error
> analysis and model cards, and shipped a tested CPU demo and least-privilege container
> while keeping the larger transformer as the benchmark quality leader.

## Two-minute technical walkthrough

1. **Data boundary:** validate the headerless CSV, remove unusable and exact duplicate
   rows, record its SHA-256, and preserve 71,117 usable records.
2. **Leakage control:** construct connected components across shared tweet IDs, exact
   text, and normalized text. A component can never cross an evaluation boundary.
3. **Classical evaluation:** compare majority, word TF-IDF, word + character TF-IDF,
   and entity-aware word + character TF-IDF on identical grouped outer folds. Select
   logistic-regression regularization only inside each training partition.
4. **Transformer evaluation:** create grouped validation data inside every outer
   training fold, train with balanced cross-entropy, and select checkpoints by
   validation macro-F1 with early stopping.
5. **Delivery:** persist preprocessing and the classical classifier together, expose
   train and predict CLIs, serve a validated local HTTP interface, and verify a
   non-root, read-only container in CI.
6. **Decision:** publish Twitter-RoBERTa as the quality winner but deploy the classical
   model in the demo because it is approximately 157 times smaller.

## Architecture discussion

```text
CSV -> validation and provenance -> connected duplicate groups
    -> grouped outer evaluation
        -> grouped inner model/checkpoint selection
        -> untouched outer-fold metrics
    -> reviewed evidence and error analysis
    -> full-data classical refit
    -> trusted artifact -> CLI / demo / container
```

The most important boundary is between selection and evaluation. The outer-fold label
must not affect feature fitting, hyperparameters, checkpoints, or stopping decisions.

## Questions recruiters are likely to ask

### Why might this score be lower than scores reported with random splits?

The scores answer different questions. A row-random split can put related augmented
records in training and testing. This project estimates generalization to unseen
connected tweet families, which is more demanding and more credible.

### Why macro-F1 instead of accuracy?

The four classes are imbalanced and `Irrelevant` is the hardest. Accuracy can improve
while minority-class behavior remains poor. Macro-F1 gives every class equal weight.

### What did entity conditioning add?

Adding target-entity context to the word + character TF-IDF model improved mean
macro-F1 from 0.5418 to 0.5686, a gain of 2.68 points on identical folds. The task asks
for sentiment about a named target, so message text alone is insufficient.

### Why does the demo use the smaller model?

The demo optimizes for inexpensive, reproducible CPU serving. Its canonical artifact
is about 3.04 MiB; Twitter-RoBERTa averages 478.92 MiB. Model quality and operational
cost are separate engineering decisions.

### What is the main model failure?

`Irrelevant` remains weakest at 0.5525 mean F1, although it gains 8.36 points over the
classical model. Major errors also occur between Neutral and Positive. The label space
mixes entity relevance with polarity, suggesting a future hierarchical objective and
a targeted label-quality audit.

### How did you prevent training-serving skew?

The classical vectorizers, entity encoder, and classifier are stored in one fitted
scikit-learn pipeline. The CLI and demo load that artifact, so inference does not
reimplement preprocessing.

### Is the confidence score reliable?

It is explicitly uncalibrated. Maximum predicted probability ranks model preference
but is not presented as a real-world probability of correctness.

### What would make this production-ready?

Production use would require an approved use case, external validation, calibrated
uncertainty, drift monitoring, privacy and retention controls, human escalation, rate
limiting, service observability, and a rollback process.

## Five-minute interview demo

1. Open the README and explain the leakage problem and headline metric.
2. Show `reports/benchmark.json` and one fold's zero-overlap diagnostics.
3. Show the transformer comparison and the `Irrelevant` error analysis.
4. Run one prediction:

   ```bash
   uv run sentiment-predict --model artifacts/sentiment_model.joblib \
     --entity Microsoft "The support update was excellent"
   ```

5. Run the browser demo if time permits:

   ```bash
   uv run sentiment-demo --model artifacts/sentiment_model.joblib
   ```

6. Finish with the quality/cost decision: transformer for benchmark quality,
   classical model for lightweight demonstration.

## Claims to avoid

- Do not call the application production-ready or claim zero technical debt.
- Do not claim calibrated probabilities, real-time monitoring, or a live public demo.
- Do not compare grouped results directly with random-split results as if they measured
  the same generalization problem.
- Do not describe the dataset as representative of people, modern language, or all
  social platforms.
