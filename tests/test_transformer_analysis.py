from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentiment_analysis.transformer_analysis import (
    analyze_transformer_evidence,
    transformer_analysis_command,
)


def _evidence() -> dict:
    labels = ["Negative", "Neutral", "Positive", "Irrelevant"]
    report = {
        label: {"f1-score": value}
        for label, value in zip(labels, [0.8, 0.5, 0.7, 0.4], strict=True)
    }
    return {
        "protocol": {"model": "example/model"},
        "dataset": {"sha256": "dataset"},
        "folds": [
            {
                "label_order": labels,
                "confusion_matrix": [
                    [8, 1, 0, 1],
                    [1, 5, 3, 1],
                    [0, 2, 7, 1],
                    [1, 2, 3, 4],
                ],
                "classification_report": report,
            }
        ],
    }


def test_transformer_analysis_pools_confusions_and_finds_weakest_class() -> None:
    report = analyze_transformer_evidence(_evidence())

    assert report["rows"] == 40
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["weakest_class"] == "Irrelevant"
    assert report["confusion_pairs"][0] == {
        "actual": "Irrelevant",
        "predicted": "Positive",
        "rows": 3,
    }
    assert report["per_class"][0]["fold_f1_standard_deviation"] == 0.0


def test_transformer_analysis_rejects_incomplete_evidence() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        analyze_transformer_evidence({"folds": []})

    evidence = _evidence()
    evidence["folds"][0]["label_order"] = []
    with pytest.raises(ValueError, match="canonical label order"):
        analyze_transformer_evidence(evidence)


def test_transformer_analysis_command_writes_traceable_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "evidence.json"
    output = tmp_path / "analysis.json"
    source.write_text(json.dumps(_evidence()), encoding="utf-8")

    transformer_analysis_command(["--input", str(source), "--output", str(output)])

    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["weakest_class"] == "Irrelevant"
    assert report["source"]["sha256"]
    assert report["dataset"] == {"sha256": "dataset"}
