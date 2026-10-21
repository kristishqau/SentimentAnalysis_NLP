# Portfolio upgrade roadmap

This roadmap tracks the remaining work required to move the project from a well-engineered classical baseline to a strong end-to-end machine-learning portfolio project. Completed work stays visible so methodological changes remain auditable.

## Milestone 1 — Trustworthy foundation

- [x] Package notebook code as an installable Python project.
- [x] Prevent tweet-ID, exact-text, and normalized near-duplicate leakage across folds.
- [x] Add CLI workflows, tests, coverage, CI, security auditing, and dependency updates.
- [x] Publish honest v1 evidence, a data card, and a model card.
- [x] Restore Python 3.10 compatibility.
- [x] Commit an exact dependency lock and verify locked installation.

Exit criterion: clean local quality suite, green Python 3.10–3.13 CI, reproducible environment.

## Milestone 2 — Defensible model comparison

- [x] Add majority, word-only, word+character, and entity-aware baselines.
- [x] Evaluate candidates with the same connected duplicate groups.
- [x] Replace iterative use of the v1 holdout with outer grouped evaluation for v2.
- [x] Report mean, standard deviation, fold results, runtime, and model size.
- [x] Fit the selected model on all usable training records for inference.

Exit criterion: checked-in benchmark evidence showing why the selected architecture wins.

## Milestone 3 — Error analysis and data quality

- [ ] Produce per-class and per-entity metrics and worst-entity slices.
- [ ] Export representative high-confidence errors for review.
- [ ] Quantify text-length and entity-frequency effects.
- [x] Add normalized near-duplicate diagnostics beyond exact matching.
- [ ] Document whether bundled dataset redistribution is permitted; otherwise provide a verified download workflow.

Exit criterion: an automated error-analysis report connected to specific next experiments.

## Milestone 4 — Modern benchmark

- [ ] Add an optional compact transformer training/evaluation extra.
- [ ] Run it against the identical grouped folds.
- [ ] Compare macro-F1, accuracy, training time, inference latency, and artifact size.
- [ ] Document the quality/cost trade-off rather than assuming the largest model is best.

Exit criterion: reproducible classical-versus-transformer evidence.

## Milestone 5 — Demonstration and delivery

- [ ] Add a safe interactive demo with entity input, examples, confidence, and limitations.
- [ ] Add a health-tested container or equally reproducible launch path.
- [ ] Publish screenshots and a concise architecture/result narrative.
- [ ] Run the final security, documentation, packaging, notebook, and repository audit.

Exit criterion: a recruiter can understand, run, and interact with the project in under five minutes.
