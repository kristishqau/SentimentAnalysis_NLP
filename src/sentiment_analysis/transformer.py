"""Optional GPU transformer benchmark using the project's exact grouped folds."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from sentiment_analysis.data import LABELS, leakage_groups, load_dataset
from sentiment_analysis.modeling import classification_metrics

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def transformer_inputs(frame: pd.DataFrame) -> list[str]:
    """Encode the target entity and message as an explicit text pair."""

    return [
        f"Target entity: {entity}\nMessage: {text}"
        for entity, text in frame[["entity", "text"]].itertuples(index=False, name=None)
    ]


def transformer_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[Any, Any]]:
    """Return the identical connected-group folds used by classical evaluation."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(frame["text"], frame["sentiment"], leakage_groups(frame)))


def _artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tokenize_batch(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    return tokenizer(batch["input_text"], truncation=True, max_length=max_length)


def run_transformer_benchmark(
    frame: pd.DataFrame,
    *,
    model_name: str = "distilbert/distilbert-base-uncased",
    n_splits: int = 5,
    epochs: float = 1.0,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fine-tune and evaluate a compact transformer on grouped outer folds."""

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the optional transformer dependencies: pip install -e '.[transformer]'"
        ) from error

    set_seed(random_state)
    texts = transformer_inputs(frame)
    labels = frame["sentiment"].map(LABEL_TO_ID).tolist()
    folds: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(
        transformer_folds(frame, n_splits=n_splits, random_state=random_state),
        start=1,
    ):
        print(f"Starting transformer fold {fold}/{n_splits}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        train_dataset = Dataset.from_dict(
            {
                "input_text": [texts[index] for index in train_indices],
                "label": [labels[index] for index in train_indices],
            }
        )
        test_dataset = Dataset.from_dict(
            {
                "input_text": [texts[index] for index in test_indices],
                "label": [labels[index] for index in test_indices],
            }
        )

        tokenize = partial(_tokenize_batch, tokenizer=tokenizer, max_length=max_length)
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["input_text"])
        test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["input_text"])
        with tempfile.TemporaryDirectory(prefix=f"sentiment-transformer-{fold}-") as temp_dir:
            arguments = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                eval_strategy="no",
                save_strategy="no",
                logging_steps=100,
                report_to="none",
                seed=random_state,
                data_seed=random_state,
                fp16=torch.cuda.is_available(),
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=train_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                processing_class=tokenizer,
            )
            fit_started = perf_counter()
            trainer.train()
            fit_seconds = perf_counter() - fit_started
            predict_started = perf_counter()
            output = trainer.predict(test_dataset)
            predict_seconds = perf_counter() - predict_started
            predicted = np.asarray(output.predictions).argmax(axis=1)
            actual_labels = [ID_TO_LABEL[index] for index in output.label_ids]
            predicted_labels = [ID_TO_LABEL[index] for index in predicted]
            metrics = classification_metrics(actual_labels, predicted_labels)
            artifact_path = Path(temp_dir) / "artifact"
            trainer.save_model(artifact_path)
            tokenizer.save_pretrained(artifact_path)
            folds.append(
                {
                    "fold": fold,
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "artifact_bytes": _artifact_size(artifact_path),
                    **metrics,
                }
            )
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_f1", "fit_seconds", "predict_seconds", "artifact_bytes"):
        values = [float(fold[metric]) for fold in folds]
        summary[metric] = {
            "mean": mean(values),
            "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "protocol": {
            "strategy": "fixed transformer evaluated on connected grouped outer folds",
            "model": model_name,
            "folds": n_splits,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "random_seed": random_state,
        },
        "folds": folds,
        "summary": summary,
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "accelerate": version("accelerate"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }


def transformer_command(argv: Sequence[str] | None = None) -> None:
    """Run the optional transformer benchmark and write JSON evidence."""

    parser = argparse.ArgumentParser(description="GPU-ready grouped transformer benchmark.")
    parser.add_argument("--data", type=Path, default=Path("data/twitter_training.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/transformer_benchmark.json"))
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    frame, profile = load_dataset(args.data)
    report = run_transformer_benchmark(
        frame,
        model_name=args.model,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    report.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": profile.to_dict(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))
