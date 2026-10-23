from __future__ import annotations

import pytest

from sentiment_analysis.transformer_study import (
    ENCODERS,
    encoder_candidates,
    merge_study_reports,
    select_validation_winner,
    tuning_candidates,
)


def test_encoder_screen_covers_all_preregistered_models_equally() -> None:
    candidates = encoder_candidates()

    assert [candidate.model for candidate in candidates] == list(ENCODERS)
    assert {candidate.learning_rate for candidate in candidates} == {2e-5}
    assert {candidate.max_length for candidate in candidates} == {128}
    assert {candidate.weight_decay for candidate in candidates} == {0.01}


def test_tuning_search_covers_bounded_coordinate_values() -> None:
    candidates = tuning_candidates("example/model")

    assert {candidate.learning_rate for candidate in candidates} == {1e-5, 2e-5, 3e-5}
    assert {candidate.max_length for candidate in candidates} == {96, 128, 192}
    assert {candidate.weight_decay for candidate in candidates} == {0.0, 0.01}
    assert len({candidate.key for candidate in candidates}) == len(candidates)


def test_validation_selection_uses_metric_and_stable_tie_break() -> None:
    records = [
        {"validation_macro_f1": 0.6, "config": {"key": "a"}},
        {"validation_macro_f1": 0.7, "config": {"key": "b"}},
        {"validation_macro_f1": 0.7, "config": {"key": "c"}},
    ]

    assert select_validation_winner(records)["config"]["key"] == "c"
    with pytest.raises(ValueError, match="At least one"):
        select_validation_winner([])


def test_merge_requires_complete_unique_comparable_folds() -> None:
    protocol = {
        "outer_folds": 2,
        "validation_splits": 2,
        "epochs_ceiling": 1.0,
        "batch_size": 8,
        "early_stopping_patience": 1,
        "selection_metric": "inner_validation_macro_f1",
        "selection_seed": 42,
        "stability_seeds": [42, 43],
        "encoder_candidates": list(ENCODERS),
        "learning_rates": [1e-5, 2e-5, 3e-5],
        "max_lengths": [96, 128, 192],
        "weight_decays": [0.0, 0.01],
        "class_weighting": "balanced",
        "seed_policy": "measure stability",
    }

    def report(fold: int) -> dict:
        return {
            "protocol": {**protocol, "completed_outer_folds": [fold]},
            "dataset": {"sha256": "same"},
            "environment": {"device": "test"},
            "folds": [
                {
                    "fold": fold,
                    "accuracy": 0.6 + fold / 100,
                    "macro_f1": 0.5 + fold / 100,
                    "selected_config": {"model": ENCODERS[fold - 1]},
                }
            ],
            "summary": {},
        }

    merged = merge_study_reports([report(2), report(1)])
    assert [fold["fold"] for fold in merged["folds"]] == [1, 2]
    assert merged["summary"]["macro_f1"]["mean"] == pytest.approx(0.515)
    assert set(merged["environments_by_fold"]) == {"1", "2"}

    with pytest.raises(ValueError, match="duplicate"):
        merge_study_reports([report(1), report(1)])
    different = report(2)
    different["dataset"] = {"sha256": "different"}
    with pytest.raises(ValueError, match="different datasets"):
        merge_study_reports([report(1), different])
