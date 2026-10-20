from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentiment_analysis.data import COLUMNS, LABELS


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str]] = []
    keywords = {
        "Negative": "awful broken frustrating",
        "Neutral": "ordinary update information",
        "Positive": "excellent delightful amazing",
        "Irrelevant": "unrelated random elsewhere",
    }
    for label_index, label in enumerate(LABELS):
        for group in range(6):
            tweet_id = str(label_index * 100 + group)
            for variant in range(2):
                rows.append(
                    (
                        tweet_id,
                        "ExampleEntity",
                        label,
                        f"{keywords[label]} sample group {group} variant {variant}",
                    )
                )
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def sample_csv(tmp_path: Path, sample_frame: pd.DataFrame) -> str:
    path = tmp_path / "sample.csv"
    sample_frame.to_csv(path, index=False, header=False)
    return str(path)
