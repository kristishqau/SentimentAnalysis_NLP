from __future__ import annotations

import pandas as pd
import pytest

from sentiment_analysis.data import normalized_group_text
from sentiment_analysis.transformer import transformer_folds, transformer_inputs


def test_transformer_inputs_include_entity_and_text(sample_frame: pd.DataFrame) -> None:
    inputs = transformer_inputs(sample_frame.iloc[:1])
    assert inputs == [
        "Target entity: ExampleEntity\nMessage: awful broken frustrating sample group 0 variant 0"
    ]


def test_transformer_folds_preserve_all_leakage_groups(sample_frame: pd.DataFrame) -> None:
    for train_indices, test_indices in transformer_folds(sample_frame, n_splits=2):
        train = sample_frame.iloc[train_indices]
        test = sample_frame.iloc[test_indices]
        assert not (set(train["tweet_id"]) & set(test["tweet_id"]))
        assert not (
            set(train["text"].map(normalized_group_text))
            & set(test["text"].map(normalized_group_text))
        )

    with pytest.raises(ValueError, match="at least 2"):
        transformer_folds(sample_frame, n_splits=1)
