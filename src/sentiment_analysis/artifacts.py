"""Versioned model artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
from sklearn.pipeline import Pipeline

SCHEMA_VERSION: Final = 1


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


def predict(path: str | Path, texts: list[str]) -> list[str]:
    """Predict sentiment labels with a trusted persisted pipeline."""

    if not texts or any(not text.strip() for text in texts):
        raise ValueError("At least one non-empty text is required")
    artifact = load_artifact(path)
    return artifact["model"].predict(texts).tolist()
