"""Leakage-safe training and evaluation for a strong linear NLP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from sentiment_analysis.data import LABELS, leakage_groups


@dataclass(frozen=True)
class TrainingResult:
    """Selected estimator and cross-validation evidence."""

    model: Pipeline
    best_parameters: dict[str, Any]
    best_cv_macro_f1: float


def build_pipeline(*, max_features: int = 40_000, random_state: int = 42) -> Pipeline:
    """Build word+character TF-IDF features and a deterministic linear classifier."""

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def select_model(
    train: pd.DataFrame,
    *,
    cv_splits: int = 3,
    max_features: int = 40_000,
    random_state: int = 42,
    jobs: int = -1,
) -> TrainingResult:
    """Select regularization strength using grouped CV on training data only."""

    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    cross_validation = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=build_pipeline(max_features=max_features, random_state=random_state),
        param_grid={"classifier__C": [0.5, 1.0, 2.0]},
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(train["text"], train["sentiment"], groups=leakage_groups(train))
    return TrainingResult(
        model=search.best_estimator_,
        best_parameters=search.best_params_,
        best_cv_macro_f1=float(search.best_score_),
    )


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a frozen model once on the untouched group holdout."""

    actual = test["sentiment"]
    predicted = model.predict(test["text"])
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro")),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "label_order": list(LABELS),
    }
