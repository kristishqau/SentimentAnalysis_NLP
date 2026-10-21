"""Dataset loading, validation, profiling, and leakage-safe splitting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

COLUMNS: Final = ("tweet_id", "entity", "sentiment", "text")
LABELS: Final = ("Negative", "Neutral", "Positive", "Irrelevant")
MIN_NORMALIZED_GROUP_LENGTH: Final = 10
_URL_PATTERN: Final = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_PATTERN: Final = re.compile(r"@\w+")
_NON_WORD_PATTERN: Final = re.compile(r"[^\w<>]+", re.UNICODE)
_WHITESPACE_PATTERN: Final = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetProfile:
    """Auditable facts about a loaded dataset."""

    rows: int
    unique_tweet_ids: int
    entities: int
    class_counts: dict[str, int]
    removed_missing_text: int
    removed_duplicate_rows: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetProfile]:
    """Load the headerless Twitter dataset and remove unusable exact duplicates.

    Similar augmented examples are intentionally retained. Exact duplicate rows and
    records without usable text are removed deterministically and reported.
    """

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(
        dataset_path,
        names=COLUMNS,
        header=None,
        dtype={"tweet_id": "string", "entity": "string", "sentiment": "string"},
    )
    unknown_labels = sorted(set(frame["sentiment"].dropna()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected sentiment labels: {unknown_labels}")

    original_rows = len(frame)
    frame = frame.dropna(subset=["tweet_id", "entity", "sentiment", "text"]).copy()
    frame["text"] = frame["text"].astype("string").str.strip()
    frame = frame.loc[frame["text"].str.len().fillna(0) > 0]
    rows_after_missing = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)

    profile = DatasetProfile(
        rows=len(frame),
        unique_tweet_ids=frame["tweet_id"].nunique(),
        entities=frame["entity"].nunique(),
        class_counts={
            label: int(count)
            for label, count in frame["sentiment"].value_counts().sort_index().items()
        },
        removed_missing_text=original_rows - rows_after_missing,
        removed_duplicate_rows=rows_after_missing - len(frame),
        sha256=_sha256(dataset_path),
    )
    return frame, profile


def normalize_text(text: str) -> str:
    """Normalize superficial variation for conservative near-duplicate grouping."""

    normalized = _URL_PATTERN.sub("<url>", text.casefold())
    normalized = _MENTION_PATTERN.sub("<user>", normalized)
    normalized = _NON_WORD_PATTERN.sub(" ", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalized_group_text(text: str) -> str:
    """Return a stable grouping value without collapsing very short generic text."""

    normalized = normalize_text(text)
    return normalized if len(normalized) >= MIN_NORMALIZED_GROUP_LENGTH else text


def leakage_groups(frame: pd.DataFrame) -> pd.Series:
    """Build connected groups over tweet IDs, exact text, and near-duplicate text.

    A simple tweet-ID group is insufficient because this dataset also repeats
    text under different IDs and with superficial formatting variation. Connected
    components prevent any known duplicate relationship from crossing evaluation or
    cross-validation boundaries.
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    row_keys: list[str] = []
    for tweet_id, text in frame[["tweet_id", "text"]].itertuples(index=False, name=None):
        id_key = f"id:{tweet_id}"
        text_hash = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        text_key = f"text:{text_hash}"
        union(id_key, text_key)
        normalized = normalized_group_text(str(text))
        normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        union(id_key, f"normalized:{normalized_hash}")
        row_keys.append(id_key)
    return pd.Series((find(key) for key in row_keys), index=frame.index, dtype="string")


def stratified_group_holdout(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one stratified holdout fold with no tweet ID shared across sets."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    required = set(COLUMNS)
    if missing := required - set(frame.columns):
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    groups = leakage_groups(frame)
    train_indices, test_indices = next(splitter.split(frame["text"], frame["sentiment"], groups))
    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)

    overlap = set(train["tweet_id"]) & set(test["tweet_id"])
    if overlap:  # Defensive invariant: a future splitter change must not leak groups.
        raise RuntimeError(f"Group leakage detected for {len(overlap)} tweet IDs")
    text_overlap = set(train["text"]) & set(test["text"])
    if text_overlap:
        raise RuntimeError(f"Text leakage detected for {len(text_overlap)} exact texts")
    normalized_overlap = set(train["text"].map(normalized_group_text)) & set(
        test["text"].map(normalized_group_text)
    )
    if normalized_overlap:
        raise RuntimeError(
            f"Near-duplicate leakage detected for {len(normalized_overlap)} normalized texts"
        )
    return train, test


def split_diagnostics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Return split facts suitable for experiment metadata."""

    train_text = set(train["text"])
    test_text = set(test["text"])
    train_normalized = set(train["text"].map(normalized_group_text))
    test_normalized = set(test["text"].map(normalized_group_text))
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_tweet_ids": train["tweet_id"].nunique(),
        "test_tweet_ids": test["tweet_id"].nunique(),
        "tweet_id_overlap": len(set(train["tweet_id"]) & set(test["tweet_id"])),
        "exact_text_overlap": len(train_text & test_text),
        "normalized_text_overlap": len(train_normalized & test_normalized),
        "train_class_counts": train["sentiment"].value_counts().sort_index().to_dict(),
        "test_class_counts": test["sentiment"].value_counts().sort_index().to_dict(),
    }
