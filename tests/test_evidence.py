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
