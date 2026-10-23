from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_checked_in_v2_evidence_is_consistent_and_leakage_free() -> None:
    report_path = Path(__file__).parents[1] / "reports" / "benchmark_v2.json"
    evidence = json.loads(report_path.read_text(encoding="utf-8"))

    assert evidence["selection"]["model_kind"] == "entity_word_char"
    assert evidence["dataset"]["rows"] == 71_117
    assert set(evidence["candidates"]) == {
        "majority",
        "word",
        "word_char",
        "entity_word_char",
    }

    selected = evidence["candidates"]["entity_word_char"]
    assert selected["summary"]["macro_f1"]["mean"] == pytest.approx(0.568596191)
    assert selected["summary"]["accuracy"]["mean"] == pytest.approx(0.581068014)
    assert all(fold["tweet_id_overlap"] == 0 for fold in selected["folds"])
    assert all(fold["exact_text_overlap"] == 0 for fold in selected["folds"])
    assert all(fold["normalized_text_overlap"] == 0 for fold in selected["folds"])
    assert all(fold["best_parameters"] == {"classifier__C": 0.5} for fold in selected["folds"])


def test_checked_in_error_analysis_matches_the_selected_model() -> None:
    report_path = Path(__file__).parents[1] / "reports" / "error_analysis.json"
    evidence = json.loads(report_path.read_text(encoding="utf-8"))

    assert evidence["protocol"]["architecture"] == "entity_word_char"
    assert evidence["protocol"]["regularization"] == 0.5
    assert evidence["overall"]["macro_f1"] == pytest.approx(0.568617099)
    assert len(evidence["high_confidence_errors"]) == 25
    assert evidence["normalized_duplicate_audit"] == {
        "normalized_variant_groups": 1540,
        "affected_rows": 3410,
        "largest_group_rows": 25,
    }


def test_checked_in_transformer_evidence_is_comparable_and_leakage_free() -> None:
    reports = Path(__file__).parents[1] / "reports"
    classical = json.loads((reports / "benchmark_v2.json").read_text(encoding="utf-8"))
    distilbert = json.loads(
        (reports / "transformer_distilbert_benchmark.json").read_text(encoding="utf-8")
    )
    twitter_roberta = json.loads(
        (reports / "transformer_benchmark.json").read_text(encoding="utf-8")
    )

    expected_test_rows = [
        fold["test_rows"] for fold in classical["candidates"]["entity_word_char"]["folds"]
    ]
    for evidence in (distilbert, twitter_roberta):
        assert evidence["dataset"] == classical["dataset"]
        assert evidence["environment"]["scikit_learn"] == "1.9.1"
        assert evidence["protocol"]["folds"] == 5
        assert evidence["protocol"]["validation_splits"] == 5
        assert evidence["protocol"]["selection_metric"] == "inner_validation_macro_f1"
        assert evidence["protocol"]["class_weighting"] == "balanced"
        assert [fold["test_rows"] for fold in evidence["folds"]] == expected_test_rows
        assert all(fold["tweet_id_overlap"] == 0 for fold in evidence["folds"])
        assert all(fold["exact_text_overlap"] == 0 for fold in evidence["folds"])
        assert all(fold["normalized_text_overlap"] == 0 for fold in evidence["folds"])
        assert all(
            fold["fit_rows"] + fold["validation_rows"] == fold["train_rows"]
            for fold in evidence["folds"]
        )

    assert distilbert["protocol"]["model"] == "distilbert/distilbert-base-uncased"
    assert distilbert["summary"]["macro_f1"]["mean"] == pytest.approx(0.603943905)
    assert twitter_roberta["protocol"]["model"] == "cardiffnlp/twitter-roberta-base"
    assert twitter_roberta["summary"]["macro_f1"]["mean"] == pytest.approx(0.635486587)
    assert twitter_roberta["summary"]["accuracy"]["mean"] == pytest.approx(0.648472080)
    assert (
        twitter_roberta["summary"]["macro_f1"]["mean"] > distilbert["summary"]["macro_f1"]["mean"]
    )
    assert (
        twitter_roberta["summary"]["macro_f1"]["mean"]
        > classical["candidates"]["entity_word_char"]["summary"]["macro_f1"]["mean"]
    )
