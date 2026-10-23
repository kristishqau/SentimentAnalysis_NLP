# Portfolio upgrade roadmap

This roadmap tracks the remaining work required to move the project from a well-engineered classical baseline to a strong end-to-end machine-learning portfolio project. Completed work stays visible so methodological changes remain auditable.

## Milestone 1 — Trustworthy foundation

- [x] Package notebook code as an installable Python project.
- [x] Prevent tweet-ID, exact-text, and normalized near-duplicate leakage across folds.
- [x] Add CLI workflows, tests, coverage, CI, security auditing, and dependency updates.
- [x] Publish honest v1 evidence, a data card, and a model card.
- [x] Support maintained Python 3.11–3.14 releases.
- [x] Commit an exact dependency lock and verify locked installation.

Exit criterion: clean local quality suite, green Python 3.11–3.14 CI, reproducible environment.

## Milestone 2 — Defensible model comparison

- [x] Add majority, word-only, word+character, and entity-aware baselines.
- [x] Evaluate candidates with the same connected duplicate groups.
- [x] Replace iterative use of the v1 holdout with outer grouped evaluation for v2.
- [x] Report mean, standard deviation, fold results, runtime, and model size.
- [x] Fit the selected model on all usable training records for inference.

Exit criterion: checked-in benchmark evidence showing why the selected architecture wins.

## Milestone 3 — Error analysis and data quality

- [x] Produce per-class and per-entity metrics and worst-entity slices.
- [x] Export one representative high-confidence error per connected family.
- [x] Quantify text-length and entity-frequency effects.
- [x] Add normalized near-duplicate diagnostics beyond exact matching.
- [x] Document the dataset's published CC0 status, provenance, integrity hashes, and rights boundary.

Exit criterion: an automated error-analysis report connected to specific next experiments.

## Milestone 4 — Modern benchmark

- [x] Add an optional compact transformer extra, grouped benchmark CLI, and Colab GPU workflow.
- [x] Run it against the identical grouped folds.
- [x] Compare macro-F1, accuracy, training time, inference latency, and artifact size.
- [x] Document the quality/cost trade-off rather than assuming the largest model is best.

Exit criterion: reproducible classical-versus-transformer evidence.

## Milestone 5 — Demonstration and delivery

- [x] Add a safe local interactive demo with entity input, examples, uncalibrated confidence, limitations, input validation, and a health endpoint.
- [x] Add a pinned, non-root container with CI-tested health and inference paths.
- [x] Publish a verified demo screenshot and concise architecture/result narrative.
- [x] Run the final security, documentation, packaging, notebook, and repository audit.

Exit criterion: a recruiter can understand, run, and interact with the project in under five minutes.

## Milestone 6 — Public portfolio deployment

- [ ] Build a self-contained image containing the trusted classical artifact.
- [ ] Verify the self-contained image without host volume mounts in CI.
- [ ] Deploy behind provider-managed TLS with health checks and resource limits.
- [ ] Add the live URL and uptime boundary to the README.

Exit criterion: a recruiter can open a public, rate-limited demo without local setup.
