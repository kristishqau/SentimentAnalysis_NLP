from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentiment_analysis.cli import predict_command, train_command


def test_training_and_prediction_commands(
    tmp_path: Path,
    sample_csv: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    train_command(
        [
            "--data",
            sample_csv,
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--holdout-folds",
            "2",
            "--cv-folds",
            "2",
            "--max-features",
            "200",
            "--jobs",
            "1",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    evidence = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert model_path.is_file()
    assert summary["tweet_id_overlap"] == 0
    assert evidence["protocol"]["holdout_used_for_selection"] is False
    assert evidence["dataset"]["sha256"]

    predict_command(["--model", str(model_path), "awful broken frustrating"])
    predictions = json.loads(capsys.readouterr().out)
    assert predictions == [{"text": "awful broken frustrating", "sentiment": "Negative"}]
