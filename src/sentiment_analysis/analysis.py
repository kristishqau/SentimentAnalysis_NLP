"""Out-of-fold diagnostics for the selected sentiment architecture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from sentiment_analysis.data import LABELS, leakage_groups, normalize_text
from sentiment_analysis.modeling import (
    MODEL_INPUT_COLUMNS,
    build_pipeline,
    classification_metrics,
)


def collect_oof_predictions(
    frame: pd.DataFrame,
    *,
    outer_splits: int = 5,
    max_features: int = 40_000,
    regularization: float = 0.5,
    random_state: int = 42,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Create one group-isolated prediction for every usable record."""

    if outer_splits < 2:
        raise ValueError("outer_splits must be at least 2")
    groups = leakage_groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    parts: list[pd.DataFrame] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame[MODEL_INPUT_COLUMNS], frame["sentiment"], groups),
        start=1,
    ):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        model = build_pipeline(
            kind="entity_word_char",
            max_features=max_features,
            random_state=random_state,
            regularization=regularization,
        )
        model.fit(train[MODEL_INPUT_COLUMNS], train["sentiment"])
        probabilities = model.predict_proba(test[MODEL_INPUT_COLUMNS])
        predicted = model.classes_[probabilities.argmax(axis=1)]
        part = test[["tweet_id", "entity", "text", "sentiment"]].copy()
        part["fold"] = fold
        part["predicted"] = predicted
        part["confidence"] = probabilities.max(axis=1)
        part["correct"] = part["sentiment"] == part["predicted"]
        parts.append(part)
        if progress:
            progress(f"Completed analysis fold {fold}/{outer_splits}")
    return pd.concat(parts).sort_index()


def _slice_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = classification_metrics(frame["sentiment"], frame["predicted"])
    return {
        "rows": len(frame),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def _group_metrics(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    return [
        {column: str(value), **_slice_metrics(group)}
        for value, group in frame.groupby(column, observed=True)
    ]


def normalized_duplicate_audit(frame: pd.DataFrame) -> dict[str, int]:
    """Quantify non-exact text families captured by normalization."""

    audit = (
        frame.assign(normalized=frame["text"].map(normalize_text))
        .loc[lambda data: data["normalized"].str.len() >= 10]
        .groupby("normalized")
        .agg(rows=("text", "size"), raw_variants=("text", "nunique"))
    )
    variants = audit.loc[audit["raw_variants"] > 1]
    return {
        "normalized_variant_groups": len(variants),
        "affected_rows": int(variants["rows"].sum()),
        "largest_group_rows": int(variants["rows"].max()) if not variants.empty else 0,
    }


def analyze_predictions(
    predictions: pd.DataFrame,
    *,
    min_entity_rows: int = 500,
    max_errors: int = 25,
) -> dict[str, Any]:
    """Build actionable aggregate and slice diagnostics from OOF predictions."""

    if min_entity_rows < 1 or max_errors < 1:
        raise ValueError("min_entity_rows and max_errors must be positive")
    analyzed = predictions.copy()
    analyzed["text_length"] = analyzed["text"].str.len()
    analyzed["length_band"] = pd.cut(
        analyzed["text_length"],
        bins=[0, 40, 80, 160, float("inf")],
        labels=["1-40", "41-80", "81-160", "161+"],
        include_lowest=True,
    )

    entity_counts = analyzed["entity"].value_counts()
    eligible_entities = entity_counts.loc[entity_counts >= min_entity_rows].index
    per_entity = _group_metrics(analyzed.loc[analyzed["entity"].isin(eligible_entities)], "entity")
    per_entity.sort(key=lambda item: (item["macro_f1"], item["entity"]))

    frequency_labels = {
        1: ["all_entities"],
        2: ["least_frequent", "most_frequent"],
        3: ["least_frequent", "mid_frequency", "most_frequent"],
        4: ["least_frequent", "low_frequency", "high_frequency", "most_frequent"],
    }
    band_count = min(4, len(entity_counts))
    if band_count == 1:
        entity_bands = pd.Series("all_entities", index=entity_counts.index)
    else:
        entity_bands = pd.qcut(
            entity_counts.rank(method="first"),
            q=band_count,
            labels=frequency_labels[band_count],
        )
    analyzed["entity_frequency_band"] = analyzed["entity"].map(entity_bands)

    errors = analyzed.loc[~analyzed["correct"]].nlargest(max_errors, "confidence")
    confusion_pairs = (
        analyzed.loc[~analyzed["correct"]]
        .groupby(["sentiment", "predicted"])
        .size()
        .sort_values(ascending=False)
    )
    overall = classification_metrics(analyzed["sentiment"], analyzed["predicted"])
    return {
        "protocol_note": (
            "Diagnostic out-of-fold predictions for the already selected architecture; "
            "do not treat this post-selection analysis as a new independent benchmark."
        ),
        "overall": overall,
        "per_entity": per_entity,
        "worst_entities": per_entity[:10],
        "by_text_length": _group_metrics(analyzed, "length_band"),
        "by_entity_frequency": _group_metrics(analyzed, "entity_frequency_band"),
        "confusion_pairs": [
            {"actual": actual, "predicted": predicted, "rows": int(count)}
            for (actual, predicted), count in confusion_pairs.items()
        ],
        "high_confidence_errors": errors[
            ["tweet_id", "entity", "text", "sentiment", "predicted", "confidence", "fold"]
        ].to_dict(orient="records"),
        "label_order": list(LABELS),
    }
