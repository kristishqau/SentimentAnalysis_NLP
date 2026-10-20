"""Leakage-safe, repeatable model comparison utilities."""

from __future__ import annotations

import io
from collections.abc import Sequence
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from sentiment_analysis.data import leakage_groups
from sentiment_analysis.modeling import (
    MODEL_INPUT_COLUMNS,
    MODEL_KINDS,
    ModelKind,
    build_pipeline,
    classification_metrics,
    select_model,
)

SUMMARY_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "fit_seconds",
    "predict_seconds",
    "artifact_bytes",
)


def _serialized_size(model: Pipeline) -> int:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.tell()


def _fit_candidate(
    train: pd.DataFrame,
    *,
    kind: ModelKind,
    inner_splits: int,
    max_features: int,
    random_state: int,
    jobs: int,
) -> tuple[Pipeline, dict[str, Any]]:
    if kind == "majority":
        model = build_pipeline(kind=kind, random_state=random_state)
        model.fit(train[MODEL_INPUT_COLUMNS], train["sentiment"])
        return model, {}

    result = select_model(
        train,
        cv_splits=inner_splits,
        max_features=max_features,
        random_state=random_state,
        jobs=jobs,
        kind=kind,
    )
    return result.model, result.best_parameters


def _summarize(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(fold[metric]) for fold in folds]
        summary[metric] = {
            "mean": mean(values),
            "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def benchmark_models(
    frame: pd.DataFrame,
    *,
    model_kinds: Sequence[ModelKind] = MODEL_KINDS,
    outer_splits: int = 5,
    inner_splits: int = 3,
    max_features: int = 40_000,
    random_state: int = 42,
    jobs: int = -1,
) -> dict[str, Any]:
    """Compare candidates with nested, connected-group cross-validation."""

    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2")
    if not model_kinds:
        raise ValueError("At least one model kind is required")
    unknown = set(model_kinds) - set(MODEL_KINDS)
    if unknown:
        raise ValueError(f"Unknown model kinds: {sorted(unknown)}")

    groups = leakage_groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_indices = list(splitter.split(frame[MODEL_INPUT_COLUMNS], frame["sentiment"], groups))
    candidates: dict[str, Any] = {}

    for kind in model_kinds:
        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_indices, test_indices) in enumerate(fold_indices, start=1):
            train = frame.iloc[train_indices].reset_index(drop=True)
            test = frame.iloc[test_indices].reset_index(drop=True)

            fit_started = perf_counter()
            model, best_parameters = _fit_candidate(
                train,
                kind=kind,
                inner_splits=inner_splits,
                max_features=max_features,
                random_state=random_state,
                jobs=jobs,
            )
            fit_seconds = perf_counter() - fit_started

            prediction_started = perf_counter()
            predictions = model.predict(test[MODEL_INPUT_COLUMNS])
            predict_seconds = perf_counter() - prediction_started
            metrics = classification_metrics(test["sentiment"], predictions)
            fold_results.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "tweet_id_overlap": len(set(train["tweet_id"]) & set(test["tweet_id"])),
                    "exact_text_overlap": len(set(train["text"]) & set(test["text"])),
                    "best_parameters": best_parameters,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "artifact_bytes": _serialized_size(model),
                    **metrics,
                }
            )
        candidates[kind] = {
            "folds": fold_results,
            "summary": _summarize(fold_results),
        }

    selected_kind = max(
        model_kinds,
        key=lambda kind: candidates[kind]["summary"]["macro_f1"]["mean"],
    )
    return {
        "protocol": {
            "strategy": "nested StratifiedGroupKFold by connected tweet_id/exact-text group",
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "random_seed": random_state,
            "selection_metric": "mean_outer_macro_f1",
            "regularization_values": [0.5, 1.0, 2.0],
        },
        "candidates": candidates,
        "selection": {"model_kind": selected_kind},
    }


def fit_selected_model(
    frame: pd.DataFrame,
    *,
    kind: ModelKind,
    inner_splits: int = 3,
    max_features: int = 40_000,
    random_state: int = 42,
    jobs: int = -1,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune and fit the selected candidate on every usable record."""

    started = perf_counter()
    model, best_parameters = _fit_candidate(
        frame,
        kind=kind,
        inner_splits=inner_splits,
        max_features=max_features,
        random_state=random_state,
        jobs=jobs,
    )
    return model, {
        "best_parameters": best_parameters,
        "fit_seconds": perf_counter() - started,
        "artifact_bytes": _serialized_size(model),
        "training_rows": len(frame),
    }
