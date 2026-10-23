# Transformer benchmark

## Decision

`cardiffnlp/twitter-roberta-base` is the quality winner. Across the same five
connected-group outer folds as the classical benchmark, it achieved **0.6355 ±
0.0140 macro-F1** and **0.6485 ± 0.0157 accuracy**.

Relative to the entity-aware classical model, this is an absolute gain of **0.0669
macro-F1** (11.76% relative) and **0.0674 accuracy**. The classical model remains the
default demo model because its approximately 3.04 MiB artifact is fast to reproduce and
deploy. The Twitter-RoBERTa fold artifact averages 478.92 MiB, approximately 157 times
larger.

## Comparable results

| Model | Macro-F1 | SD | Accuracy | SD | Mean artifact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entity + word + character TF-IDF | 0.5686 | 0.0090 | 0.5811 | 0.0095 | 3.04 MiB |
| DistilBERT | 0.6039 | 0.0093 | 0.6187 | 0.0101 | 256.12 MiB |
| Twitter-RoBERTa | **0.6355** | **0.0140** | **0.6485** | **0.0157** | 478.92 MiB |

Mean per-class F1 shows that the quality gain is not limited to the largest class:

| Class | Classical | DistilBERT | Twitter-RoBERTa | Winner gain vs classical |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.6648 | 0.7017 | **0.7333** | +0.0685 |
| Neutral | 0.5540 | 0.5746 | **0.5948** | +0.0408 |
| Positive | 0.5867 | 0.6338 | **0.6613** | +0.0746 |
| Irrelevant | 0.4689 | 0.5057 | **0.5525** | +0.0836 |

`Irrelevant` remains the weakest class, but it receives the largest absolute class-level
improvement.

## Protocol and integrity

Both transformers use the canonical outer fold test sizes of 14,210, 14,227, 14,220,
14,250, and 14,210 records. Every fold records zero shared tweet IDs, exact texts, and
normalized texts. The dataset SHA-256 matches the classical evidence.

Within each outer training partition, a five-way connected-group split creates an inner
validation partition. Models train for at most three epochs with inverse-frequency
cross-entropy, select checkpoints by validation macro-F1, and stop early. The outer test
partition is never used for checkpoint selection. Four of five Twitter-RoBERTa folds
selected epoch 2; one selected epoch 3.

The run used Python 3.13.15, scikit-learn 1.9.1, PyTorch 2.11.0 with CUDA 13.0,
Transformers 5.20.1, Accelerate 1.15.0, and a Tesla T4.

## Cost and deployment trade-off

Twitter-RoBERTa averaged 556.9 seconds of training and 18.0 seconds of batched prediction
per outer fold on the T4. Its five training phases totaled 46.4 minutes, excluding data
preparation and notebook overhead. DistilBERT totaled 24.5 minutes and averaged 294.5
seconds per fold.

These timings are hardware-specific and should not be treated as a CPU/GPU speed contest.
They demonstrate the project decision instead: Twitter-RoBERTa provides the best quality,
while the classical pipeline provides a much smaller operational baseline. A public demo
can lead with the classical artifact and present the transformer as the audited quality
ceiling until transformer serving cost is intentionally accepted.

Machine-readable evidence is stored in [`transformer_benchmark.json`](transformer_benchmark.json)
and [`transformer_distilbert_benchmark.json`](transformer_distilbert_benchmark.json).
