"""Command-line interfaces for reproducible training and prediction."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sentiment_analysis import __version__
from sentiment_analysis.artifacts import predict, save_artifact
from sentiment_analysis.data import load_dataset, split_diagnostics, stratified_group_holdout
from sentiment_analysis.modeling import evaluate, select_model


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-safe tweet sentiment classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/twitter_training.csv"),
        help="Headerless four-column training CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/sentiment_model.joblib"),
        help="Destination for the trusted joblib pipeline.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Destination for machine-readable experiment evidence.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-folds", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=40_000)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser


def _predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict sentiment with a trusted local model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/sentiment_model.joblib"),
    )
    parser.add_argument(
        "--entity",
        default="Unknown",
        help="Entity targeted by every supplied text (unknown entities are supported).",
    )
    parser.add_argument("text", nargs="+", help="One or more texts to classify.")
    return parser


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the training protocol and return its serializable evidence."""

    frame, profile = load_dataset(args.data)
    train, holdout = stratified_group_holdout(
        frame,
        n_splits=args.holdout_folds,
        random_state=args.seed,
    )
    selected = select_model(
        train,
        cv_splits=args.cv_folds,
        max_features=args.max_features,
        random_state=args.seed,
        jobs=args.jobs,
    )
    holdout_metrics = evaluate(selected.model, holdout)
    evidence: dict[str, Any] = {
        "project_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "random_seed": args.seed,
            "holdout_strategy": (
                "first fold of shuffled StratifiedGroupKFold by connected tweet_id/exact-text group"
            ),
            "holdout_folds": args.holdout_folds,
            "selection_strategy": (
                "GridSearchCV with StratifiedGroupKFold by connected tweet_id/exact-text group"
            ),
            "cross_validation_folds": args.cv_folds,
            "selection_metric": "macro_f1",
            "holdout_used_for_selection": False,
        },
        "dataset": profile.to_dict(),
        "split": split_diagnostics(train, holdout),
        "selection": {
            "architecture": "entity_word_char",
            "best_parameters": selected.best_parameters,
            "best_cv_macro_f1": selected.best_cv_macro_f1,
        },
        "holdout": holdout_metrics,
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    save_artifact(args.model_output, model=selected.model, metadata=evidence)
    return evidence


def train_command(argv: Sequence[str] | None = None) -> None:
    """Console entry point for model training."""

    args = _train_parser().parse_args(argv)
    evidence = run_training(args)
    summary = {
        "model": str(args.model_output),
        "metrics": str(args.metrics_output),
        "cv_macro_f1": evidence["selection"]["best_cv_macro_f1"],
        "holdout_macro_f1": evidence["holdout"]["macro_f1"],
        "holdout_accuracy": evidence["holdout"]["accuracy"],
        "tweet_id_overlap": evidence["split"]["tweet_id_overlap"],
    }
    print(json.dumps(summary, indent=2))


def predict_command(argv: Sequence[str] | None = None) -> None:
    """Console entry point for batch prediction."""

    args = _predict_parser().parse_args(argv)
    entities = [args.entity] * len(args.text)
    labels = predict(args.model, args.text, entities=entities)
    print(
        json.dumps(
            [
                {"entity": entity, "text": text, "sentiment": label}
                for entity, text, label in zip(entities, args.text, labels, strict=True)
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
