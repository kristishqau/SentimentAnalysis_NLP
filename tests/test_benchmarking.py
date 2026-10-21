from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_analysis.artifacts import load_artifact
from sentiment_analysis.benchmark_cli import run_benchmark
from sentiment_analysis.benchmarking import benchmark_models


def test_benchmark_compares_models_without_group_leakage(sample_frame) -> None:
    evidence = benchmark_models(
        sample_frame,
        model_kinds=["majority", "word"],
        outer_splits=2,
        inner_splits=2,
        max_features=100,
        jobs=1,
    )

    assert evidence["selection"]["model_kind"] == "word"
    assert set(evidence["candidates"]) == {"majority", "word"}
    for candidate in evidence["candidates"].values():
        assert len(candidate["folds"]) == 2
        assert candidate["summary"]["macro_f1"]["standard_deviation"] >= 0
        assert all(fold["tweet_id_overlap"] == 0 for fold in candidate["folds"])
        assert all(fold["exact_text_overlap"] == 0 for fold in candidate["folds"])
        assert all(fold["normalized_text_overlap"] == 0 for fold in candidate["folds"])


def test_benchmark_rejects_invalid_configuration(sample_frame) -> None:
    for options, message in [
        ({"outer_splits": 1}, "at least 2"),
        ({"model_kinds": []}, "At least one"),
        ({"model_kinds": ["invalid"]}, "Unknown model"),
    ]:
        try:
            benchmark_models(sample_frame, **options)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Expected invalid benchmark configuration to fail")


def test_benchmark_workflow_writes_report_and_model(
    sample_csv: str,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "benchmark.json"
    model_path = tmp_path / "model.joblib"
    args = argparse.Namespace(
        data=Path(sample_csv),
        report_output=report_path,
        model_output=model_path,
        outer_folds=2,
        inner_folds=2,
        max_features=100,
        seed=42,
        jobs=1,
        models=["majority"],
    )

    evidence = run_benchmark(args)

    assert report_path.is_file()
    assert model_path.is_file()
    assert evidence["selection"]["model_kind"] == "majority"
    assert evidence["final_model"]["training_rows"] == 48
    assert load_artifact(model_path)["metadata"]["dataset"]["rows"] == 48
