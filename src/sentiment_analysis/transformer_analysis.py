"""Aggregate error diagnostics from canonical transformer fold evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from sentiment_analysis.data import LABELS


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze_transformer_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Pool fold confusion matrices and expose weak-class diagnostics."""

    folds = evidence.get("folds", [])
    if not folds:
        raise ValueError("Transformer evidence must contain at least one fold")
    expected_labels = list(LABELS)
    if any(fold.get("label_order") != expected_labels for fold in folds):
        raise ValueError("Every fold must use the canonical label order")

    size = len(expected_labels)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for fold in folds:
        matrix = fold.get("confusion_matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != size
            or any(not isinstance(row, list) or len(row) != size for row in matrix)
        ):
            raise ValueError("Every fold must contain a square canonical confusion matrix")
        for actual in range(size):
            for predicted in range(size):
                confusion[actual][predicted] += int(matrix[actual][predicted])

    per_class: list[dict[str, Any]] = []
    for index, label in enumerate(expected_labels):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_rows = sum(row[index] for row in confusion)
        precision = _safe_divide(true_positive, predicted_rows)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        fold_f1 = [float(fold["classification_report"][label]["f1-score"]) for fold in folds]
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "fold_f1_mean": mean(fold_f1),
                "fold_f1_standard_deviation": stdev(fold_f1) if len(fold_f1) > 1 else 0.0,
                "fold_f1_min": min(fold_f1),
                "fold_f1_max": max(fold_f1),
            }
        )

    confusions = sorted(
        (
            {
                "actual": actual_label,
                "predicted": predicted_label,
                "rows": confusion[actual_index][predicted_index],
            }
            for actual_index, actual_label in enumerate(expected_labels)
            for predicted_index, predicted_label in enumerate(expected_labels)
            if actual_index != predicted_index
        ),
        key=lambda item: (-item["rows"], item["actual"], item["predicted"]),
    )
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(size))
    weakest = min(per_class, key=lambda item: (item["f1"], item["label"]))
    return {
        "protocol_note": (
            "Post-selection diagnostics pooled from group-isolated outer-fold predictions; "
            "this report does not constitute an additional benchmark."
        ),
        "model": evidence["protocol"]["model"],
        "folds": len(folds),
        "rows": total,
        "accuracy": _safe_divide(correct, total),
        "macro_f1": mean(item["f1"] for item in per_class),
        "weakest_class": weakest["label"],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_pairs": confusions,
        "label_order": expected_labels,
    }


def transformer_analysis_command(argv: Sequence[str] | None = None) -> None:
    """Create machine-readable transformer diagnostics from benchmark evidence."""

    parser = argparse.ArgumentParser(description="Analyze canonical transformer evidence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/transformer_error_analysis.json"),
    )
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    evidence = json.loads(raw)
    report = analyze_transformer_evidence(evidence)
    report.update(
        {
            "source": {
                "path": args.input.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "dataset": evidence.get("dataset"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": report["model"],
                "macro_f1": report["macro_f1"],
                "weakest_class": report["weakest_class"],
            },
            indent=2,
        )
    )
