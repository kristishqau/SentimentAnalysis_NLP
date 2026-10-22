from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pytest

from sentiment_analysis.data import normalized_group_text
from sentiment_analysis.transformer import (
    _reference_test_rows,
    balanced_label_weights,
    grouped_validation_split,
    run_transformer_benchmark,
    transformer_folds,
    transformer_inputs,
)


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


def test_grouped_validation_split_stays_inside_outer_training_data(
    sample_frame: pd.DataFrame,
) -> None:
    outer_train, _ = transformer_folds(sample_frame, n_splits=2)[0]
    fit_indices, validation_indices = grouped_validation_split(
        sample_frame,
        outer_train,
        n_splits=2,
    )
    fit = sample_frame.iloc[fit_indices]
    validation = sample_frame.iloc[validation_indices]

    assert set(fit_indices) | set(validation_indices) == set(outer_train)
    assert not (set(fit_indices) & set(validation_indices))
    assert not (set(fit["tweet_id"]) & set(validation["tweet_id"]))
    assert not (
        set(fit["text"].map(normalized_group_text))
        & set(validation["text"].map(normalized_group_text))
    )


def test_balanced_label_weights_emphasize_the_rarest_class() -> None:
    weights = balanced_label_weights(
        ["Negative"] * 4 + ["Neutral"] * 3 + ["Positive"] * 2 + ["Irrelevant"]
    )

    assert list(weights) == ["Negative", "Neutral", "Positive", "Irrelevant"]
    assert weights["Irrelevant"] > weights["Positive"] > weights["Neutral"]
    with pytest.raises(ValueError, match="missing from training data"):
        balanced_label_weights(["Negative", "Neutral"])


def test_transformer_benchmark_rejects_different_reference_folds(
    sample_frame: pd.DataFrame,
) -> None:
    with pytest.raises(RuntimeError, match="do not match the reference"):
        run_transformer_benchmark(sample_frame, n_splits=2, expected_test_rows=[1, 1])


def test_reference_requires_matching_dataset_folds_and_sklearn(tmp_path: Path) -> None:
    reference = {
        "dataset": {"sha256": "dataset-hash"},
        "protocol": {"outer_splits": 2},
        "environment": {"scikit_learn": version("scikit-learn")},
        "selection": {"model_kind": "selected"},
        "candidates": {
            "selected": {"folds": [{"test_rows": 4}, {"test_rows": 6}]},
        },
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    assert _reference_test_rows(path, dataset_sha256="dataset-hash", n_splits=2) == [4, 6]

    with pytest.raises(ValueError, match="different dataset"):
        _reference_test_rows(path, dataset_sha256="other", n_splits=2)
    with pytest.raises(ValueError, match="has 2 folds, not 3"):
        _reference_test_rows(path, dataset_sha256="dataset-hash", n_splits=3)
