"""CLI for reproducible model benchmarking and final model fitting."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from sentiment_analysis import __version__
from sentiment_analysis.artifacts import save_artifact
from sentiment_analysis.benchmarking import benchmark_models, fit_selected_model
from sentiment_analysis.data import load_dataset
from sentiment_analysis.modeling import MODEL_KINDS, ModelKind


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark sentiment architectures with nested grouped cross-validation."
    )
    parser.add_argument("--data", type=Path, default=Path("data/twitter_training.csv"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/benchmark.json"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/sentiment_model.joblib"),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KINDS,
        default=list(MODEL_KINDS),
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the benchmark, fit its winner, and persist all evidence."""

    frame, profile = load_dataset(args.data)
    model_kinds = cast(list[ModelKind], args.models)
    evidence = benchmark_models(
        frame,
        model_kinds=model_kinds,
        outer_splits=args.outer_folds,
        inner_splits=args.inner_folds,
        max_features=args.max_features,
        random_state=args.seed,
        jobs=args.jobs,
    )
    selected_kind = cast(ModelKind, evidence["selection"]["model_kind"])
    model, final_model = fit_selected_model(
        frame,
        kind=selected_kind,
        inner_splits=args.inner_folds,
        max_features=args.max_features,
        random_state=args.seed,
        jobs=args.jobs,
    )
    evidence.update(
        {
            "project_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": profile.to_dict(),
            "final_model": final_model,
            "environment": {
                "python": platform.python_version(),
                "pandas": version("pandas"),
                "scikit_learn": version("scikit-learn"),
            },
        }
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    save_artifact(args.model_output, model=model, metadata=evidence)
    return evidence


def benchmark_command(argv: Sequence[str] | None = None) -> None:
    """Console entry point for the benchmark workflow."""

    args = _parser().parse_args(argv)
    evidence = run_benchmark(args)
    selected_kind = evidence["selection"]["model_kind"]
    selected = evidence["candidates"][selected_kind]["summary"]
    print(
        json.dumps(
            {
                "selected_model": selected_kind,
                "outer_macro_f1": selected["macro_f1"],
                "outer_accuracy": selected["accuracy"],
                "report": str(args.report_output),
                "model": str(args.model_output),
            },
            indent=2,
        )
    )
