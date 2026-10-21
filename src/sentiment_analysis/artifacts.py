"""Versioned model artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

SCHEMA_VERSION: Final = 2


def save_artifact(
    path: str | Path,
    *,
    model: Pipeline,
    metadata: dict[str, Any],
) -> None:
    """Persist preprocessing and classifier together with experiment metadata."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"schema_version": SCHEMA_VERSION, "model": model, "metadata": metadata},
        artifact_path,
    )


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local artifact.

    Joblib artifacts can execute code while loading. Never load a model from an
    untrusted source.
    """

    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed model artifact")
    if not isinstance(artifact.get("model"), Pipeline):
        raise ValueError("Artifact does not contain a scikit-learn Pipeline")
    if not isinstance(artifact.get("metadata"), dict):
        raise ValueError("Artifact metadata is missing or malformed")
    return artifact


def predict(
    path: str | Path,
    texts: list[str],
    *,
    entities: list[str] | None = None,
) -> list[str]:
    """Predict sentiment labels with a trusted persisted pipeline."""

    inputs = _prediction_frame(texts, entities)
    artifact = load_artifact(path)
    return artifact["model"].predict(inputs).tolist()


def predict_with_confidence(
    path: str | Path,
    texts: list[str],
    *,
    entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Predict labels and uncalibrated maximum class probabilities."""

    inputs = _prediction_frame(texts, entities)
    model = load_artifact(path)["model"]
    probabilities = model.predict_proba(inputs)
    indices = probabilities.argmax(axis=1)
    return [
        {
            "entity": str(inputs.iloc[index]["entity"]),
            "text": str(inputs.iloc[index]["text"]),
            "sentiment": str(model.classes_[class_index]),
            "confidence": float(probabilities[index, class_index]),
        }
        for index, class_index in enumerate(indices)
    ]


def _prediction_frame(texts: list[str], entities: list[str] | None) -> pd.DataFrame:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("At least one non-empty text is required")
    if entities is None:
        entities = ["Unknown"] * len(texts)
    if len(entities) != len(texts):
        raise ValueError("entities and texts must have the same length")
    return pd.DataFrame({"entity": entities, "text": texts})
