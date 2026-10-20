from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentiment_analysis.data import (
    COLUMNS,
    load_dataset,
    split_diagnostics,
    stratified_group_holdout,
)


def test_load_dataset_cleans_and_profiles(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    rows = [
        ("1", "Entity", "Positive", "great result"),
        ("1", "Entity", "Positive", "great result"),
        ("2", "Entity", "Negative", None),
        ("3", "Other", "Neutral", "   "),
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, header=False)

    frame, profile = load_dataset(path)

    assert len(frame) == 1
    assert profile.rows == 1
    assert profile.removed_missing_text == 2
    assert profile.removed_duplicate_rows == 1
    assert profile.class_counts == {"Positive": 1}
    assert len(profile.sha256) == 64
    assert profile.to_dict()["unique_tweet_ids"] == 1


def test_load_dataset_rejects_missing_file_and_unknown_label(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.csv")

    path = tmp_path / "invalid.csv"
    pd.DataFrame([("1", "Entity", "Surprised", "hello")], columns=COLUMNS).to_csv(
        path, index=False, header=False
    )
    with pytest.raises(ValueError, match="Unexpected sentiment labels"):
        load_dataset(path)


def test_group_holdout_has_no_id_leakage(sample_frame: pd.DataFrame) -> None:
    train, test = stratified_group_holdout(sample_frame, n_splits=2, random_state=7)
    diagnostics = split_diagnostics(train, test)

    assert diagnostics["tweet_id_overlap"] == 0
    assert diagnostics["exact_text_overlap"] == 0
    assert diagnostics["train_rows"] + diagnostics["test_rows"] == len(sample_frame)
    assert set(train["sentiment"]) == set(test["sentiment"])


def test_group_holdout_validates_arguments(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        stratified_group_holdout(sample_frame, n_splits=1)
    with pytest.raises(ValueError, match="missing required columns"):
        stratified_group_holdout(sample_frame.drop(columns="entity"))
