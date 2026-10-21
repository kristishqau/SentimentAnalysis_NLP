from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentiment_analysis.analysis import (
    analyze_predictions,
    collect_oof_predictions,
    normalized_duplicate_audit,
)
from sentiment_analysis.analysis_cli import analysis_command


def test_oof_error_analysis_covers_every_row(sample_frame) -> None:
    progress: list[str] = []
    predictions = collect_oof_predictions(
        sample_frame,
        outer_splits=2,
        max_features=100,
        progress=progress.append,
    )
    report = analyze_predictions(predictions, min_entity_rows=1, max_errors=5)

    assert len(predictions) == len(sample_frame)
    assert predictions.index.is_unique
    assert set(predictions["fold"]) == {1, 2}
    assert report["overall"]["macro_f1"] == pytest.approx(1.0)
    assert report["per_entity"][0]["entity"] == "ExampleEntity"
    assert progress[-1] == "Completed analysis fold 2/2"


def test_analysis_validates_arguments_and_audits_variants(sample_frame) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        collect_oof_predictions(sample_frame, outer_splits=1)

    predictions = sample_frame.rename(columns={"sentiment": "predicted"}).assign(
        sentiment=sample_frame["sentiment"],
        confidence=1.0,
        correct=True,
        fold=1,
    )
    with pytest.raises(ValueError, match="positive"):
        analyze_predictions(predictions, min_entity_rows=0)

    variants = sample_frame.iloc[:2].copy()
    variants.loc[1, "text"] = variants.loc[0, "text"].upper() + "!!!"
    audit = normalized_duplicate_audit(variants)
    assert audit["normalized_variant_groups"] == 1
    assert audit["affected_rows"] == 2


def test_analysis_command_writes_machine_readable_report(
    sample_csv: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "analysis.json"
    analysis_command(
        [
            "--data",
            sample_csv,
            "--output",
            str(output),
            "--folds",
            "2",
            "--max-features",
            "100",
            "--min-entity-rows",
            "1",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["rows"] == 48
    assert report["protocol"]["architecture"] == "entity_word_char"
    assert report["dataset"]["sha256"]
