from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from sentiment_analysis.artifacts import load_artifact, predict, save_artifact
from sentiment_analysis.data import stratified_group_holdout
from sentiment_analysis.modeling import MODEL_KINDS, build_pipeline, evaluate, select_model


def test_model_selection_evaluation_and_round_trip(
    tmp_path: Path, sample_frame: pd.DataFrame
) -> None:
    train, test = stratified_group_holdout(sample_frame, n_splits=2, random_state=42)
    result = select_model(
        train,
        cv_splits=2,
        max_features=200,
        random_state=42,
        jobs=1,
    )
    metrics = evaluate(result.model, test)

    assert result.best_parameters["classifier__C"] in {0.5, 1.0, 2.0}
    assert 0 <= result.best_cv_macro_f1 <= 1
    assert metrics["macro_f1"] >= 0.9
    assert len(metrics["confusion_matrix"]) == 4
    assert metrics["label_order"] == ["Negative", "Neutral", "Positive", "Irrelevant"]

    path = tmp_path / "model.joblib"
    save_artifact(path, model=result.model, metadata={"purpose": "test"})
    artifact = load_artifact(path)

    assert artifact["metadata"] == {"purpose": "test"}
    assert (
        predict(
            path,
            ["excellent delightful amazing"],
            entities=["ExampleEntity"],
        )[0]
        == "Positive"
    )


def test_artifact_validation(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.joblib"
    joblib.dump({"schema_version": 999}, malformed)
    with pytest.raises(ValueError, match="Unsupported or malformed"):
        load_artifact(malformed)
    with pytest.raises(ValueError, match="non-empty text"):
        predict(malformed, [])
    with pytest.raises(ValueError, match="same length"):
        predict(malformed, ["valid text"], entities=[])


def test_all_model_kinds_build_and_unknown_kind_is_rejected() -> None:
    for kind in MODEL_KINDS:
        assert build_pipeline(kind=kind)
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_pipeline(kind="unsupported")  # type: ignore[arg-type]
