"""Leakage-safe training and evaluation for a strong linear NLP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sentiment_analysis.data import LABELS, leakage_groups

ModelKind = Literal["majority", "word", "word_char", "entity_word_char"]
MODEL_KINDS: Final[tuple[ModelKind, ...]] = (
    "majority",
    "word",
    "word_char",
    "entity_word_char",
)
MODEL_INPUT_COLUMNS: Final = ["entity", "text"]


@dataclass(frozen=True)
class TrainingResult:
    """Selected estimator and cross-validation evidence."""

    model: Pipeline
    best_parameters: dict[str, Any]
    best_cv_macro_f1: float


def build_pipeline(
    *,
    kind: ModelKind = "entity_word_char",
    max_features: int = 40_000,
    random_state: int = 42,
    regularization: float = 1.0,
) -> Pipeline:
    """Build one of the comparable baseline or entity-aware pipelines."""

    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind: {kind}")
    if kind == "majority":
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])

    transformers: list[tuple[str, Any, str | list[str]]] = [
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
            "text",
        )
    ]
    if kind in {"word_char", "entity_word_char"}:
        transformers.append(
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
                "text",
            )
        )
    if kind == "entity_word_char":
        transformers.append(
            (
                "entity",
                OneHotEncoder(handle_unknown="ignore"),
                ["entity"],
            )
        )

    features = ColumnTransformer(
        transformers,
        sparse_threshold=1.0,
    )
    classifier = LogisticRegression(
        C=regularization,
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
    kind: ModelKind = "entity_word_char",
) -> TrainingResult:
    """Select regularization strength using grouped CV on training data only."""

    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if kind == "majority":
        raise ValueError("The majority baseline has no regularization to select")
    cross_validation = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=build_pipeline(
            kind=kind,
            max_features=max_features,
            random_state=random_state,
        ),
        param_grid={"classifier__C": [0.5, 1.0, 2.0]},
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(
        train[MODEL_INPUT_COLUMNS],
        train["sentiment"],
        groups=leakage_groups(train),
    )
    return TrainingResult(
        model=search.best_estimator_,
        best_parameters=search.best_params_,
        best_cv_macro_f1=float(search.best_score_),
    )


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a frozen model once on the untouched group holdout."""

    actual = test["sentiment"]
    predicted = model.predict(test[MODEL_INPUT_COLUMNS])
    return classification_metrics(actual, predicted)


def classification_metrics(actual: Any, predicted: Any) -> dict[str, Any]:
    """Return the project's consistent multiclass metric suite."""

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
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
