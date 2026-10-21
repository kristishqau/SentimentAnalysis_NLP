# V2 model benchmark

## Decision

The selected production candidate is `entity_word_char`: word and character TF-IDF,
one-hot entity context, and class-balanced logistic regression. Across five outer
connected-group folds it achieved **0.5713 ± 0.0084 macro-F1** and
**0.5834 ± 0.0088 accuracy**.

The complete fold-level evidence is in [`benchmark_v2.json`](benchmark_v2.json).
Every outer fold has zero shared tweet IDs and zero shared exact texts between its
training and test partitions.

## Candidate comparison

| Candidate | Mean macro-F1 | SD | Mean accuracy | Typical compressed model size |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 0.1164 | 0.0000 | 0.3033 | <0.01 MB |
| Word TF-IDF | 0.5363 | 0.0074 | 0.5547 | 1.58 MB |
| Word + character TF-IDF | 0.5458 | 0.0092 | 0.5628 | 3.04 MB |
| Word + character + entity | **0.5713** | **0.0084** | **0.5834** | 3.04 MB |

Entity context adds 2.55 macro-F1 points over the otherwise identical word+character
model. All five entity-aware outer folds selected `C=0.5` using only their inner
grouped folds.

Mean entity-aware per-class F1 across the outer folds was:

- Negative: 0.6672
- Positive: 0.5913
- Neutral: 0.5529
- Irrelevant: 0.4739

`Irrelevant` remains the weakest class and is a priority for error analysis.

## Protocol

The benchmark uses five-fold outer `StratifiedGroupKFold`. Within each non-trivial
candidate and outer fold, three-fold grouped cross-validation chooses logistic
regression `C` from `0.5`, `1.0`, and `2.0`. Groups are connected components over
both tweet ID and exact-text hash, preventing either duplicate relationship from
crossing a fold boundary. Macro-F1 is the selection metric because accuracy can hide
poor behavior on the smaller classes.

After the comparison, the winner is tuned with grouped cross-validation and fitted
on all 71,117 usable records for inference. That final generated artifact is excluded
from Git and can be recreated with `sentiment-benchmark`.

## Runtime caveat

The raw report preserves measured elapsed wall time. The computer slept during
entity-aware fold 1, inflating that fold to 37,871 seconds. Its other four folds took
405–429 seconds (median 416 seconds), and their scores are unaffected. Therefore the
entity model's stored mean wall time is not suitable for comparison. An uninterrupted
full benchmark took approximately one hour of active wall time on the reference
Windows laptop; runtime will vary substantially by CPU, memory, and job count.
