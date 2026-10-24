# Project status

## Final decision

This repository is ready to present as a machine-learning engineering portfolio
project. It provides a reproducible path from raw data to evaluation, artifact
creation, command-line inference, a browser demo, and a hardened container.

The validated headline result is Twitter-RoBERTa at **0.6355 ± 0.0140 macro-F1** and
**0.6485 ± 0.0157 accuracy** over five connected-group outer folds. The lightweight
entity-aware classical model scores **0.5686 ± 0.0090 macro-F1** and powers the demo.
The transformer improves macro-F1 by **0.0669 absolute points**, or **11.76% relative**.

## Why the result is credible

- The 71,117 usable rows are grouped by connected tweet ID, exact text, and normalized
  text relationships before splitting.
- Every published fold has zero tweet-ID, exact-text, and normalized-text overlap.
- Model and checkpoint selection happen on grouped training/validation partitions;
  outer test partitions are used only for evaluation.
- Evidence includes dataset hashes, fold metrics, uncertainty, class metrics, runtime,
  artifact size, package versions, and leakage diagnostics.
- Quality gates include 30 tests, more than 90% measured coverage, lint and format
  checks, Python 3.11–3.14 CI, dependency auditing, container health checks, and an
  exact dependency lock.

## Product boundary

The project deliberately has two model tiers:

- **Research quality leader:** Twitter-RoBERTa, used for the strongest validated
  benchmark and class-level evidence.
- **Deployment baseline:** entity + word/character TF-IDF + balanced logistic
  regression, used by the CPU-friendly demo and self-contained container.

This is an engineering trade-off. The transformer artifact is about 157 times larger,
while the classical artifact is approximately 3.04 MiB in the canonical benchmark and
can be reproduced locally without a GPU.

## Known limitations

`Irrelevant` remains the hardest class at 0.5525 mean F1 for Twitter-RoBERTa. The
dataset is historical English social-media text for 32 entities and contains ambiguous
or noisy labels. Confidence is explicitly uncalibrated. The project is unsuitable for
moderation, decisions about people, or inference about identity, intent, or mental
state.

## Further development

Useful extensions include calibrated confidence, a hierarchical relevance-then-
polarity objective, external-dataset validation, and managed public hosting. They are
opportunities to extend an already complete portfolio project, not dependencies of the
published result.
