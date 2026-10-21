"""CLI for leakage-safe out-of-fold error analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from sentiment_analysis.analysis import (
    analyze_predictions,
    collect_oof_predictions,
    normalized_duplicate_audit,
)
from sentiment_analysis.data import load_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate out-of-fold error diagnostics.")
    parser.add_argument("--data", type=Path, default=Path("data/twitter_training.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/error_analysis.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=40_000)
    parser.add_argument("--regularization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-entity-rows", type=int, default=500)
    parser.add_argument("--max-errors", type=int, default=25)
    return parser


def analysis_command(argv: Sequence[str] | None = None) -> None:
    """Console entry point for out-of-fold diagnostics."""

    args = _parser().parse_args(argv)
    frame, profile = load_dataset(args.data)
    predictions = collect_oof_predictions(
        frame,
        outer_splits=args.folds,
        max_features=args.max_features,
        regularization=args.regularization,
        random_state=args.seed,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    report = analyze_predictions(
        predictions,
        min_entity_rows=args.min_entity_rows,
        max_errors=args.max_errors,
    )
    report.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": profile.to_dict(),
            "normalized_duplicate_audit": normalized_duplicate_audit(frame),
            "protocol": {
                "folds": args.folds,
                "random_seed": args.seed,
                "regularization": args.regularization,
                "architecture": "entity_word_char",
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": profile.rows,
                "macro_f1": report["overall"]["macro_f1"],
                "errors_reviewed": len(report["high_confidence_errors"]),
            },
            indent=2,
        )
    )
