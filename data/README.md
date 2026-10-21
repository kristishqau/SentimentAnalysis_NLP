# Dataset provenance and license

These files are the **Twitter Entity Sentiment Analysis** dataset attributed to the
Kaggle publisher `passionate-nlp` / `jp797498e`:

- [Original Kaggle dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Maintained Kaggle mirror and dataset card](https://huggingface.co/datasets/haoxianc/jp797498e_twitter-entity-sentiment-analysis)

The mirror records the source dataset as **CC0 1.0 / Public Domain** and preserves
the original attribution. This was checked on 2026-10-21. The CSVs remain in this
repository for exact experiment reproduction under that published metadata.

## Integrity

- `twitter_training.csv`:
  `68ebaee4f44bda5de49b96b5e5630184f1af894970b994aa6bb0aa5e1dbfe9a0`
- `twitter_validation.csv`:
  `f06b0791cca4c977bdf5093678a704ba97fcb7fc592c963c45424346f80136f1`

Verify on PowerShell:

```powershell
Get-FileHash data\twitter_training.csv -Algorithm SHA256
Get-FileHash data\twitter_validation.csv -Algorithm SHA256
```

## Important boundary

The repository's MIT license applies only to original code and documentation. The
dataset's stated CC0 status is separate. Because the files contain historical social-
media text, handles, offensive language, and potentially third-party content, users
must still consider privacy, platform terms, removal requests, and applicable law.
Public availability is not a claim that every downstream use is appropriate.
