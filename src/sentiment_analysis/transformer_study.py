"""Nested, GPU-oriented transformer encoder and hyperparameter study."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any

from sentiment_analysis.data import LABELS, load_dataset, split_diagnostics
from sentiment_analysis.modeling import classification_metrics
from sentiment_analysis.transformer import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    _artifact_size,
    _reference_test_rows,
    _tokenize_batch,
    balanced_label_weights,
    grouped_validation_split,
    transformer_folds,
    transformer_inputs,
)

ENCODERS = (
    "distilbert/distilbert-base-uncased",
    "cardiffnlp/twitter-roberta-base",
    "distilbert/distilroberta-base",
    "microsoft/deberta-v3-small",
)
STABILITY_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class StudyConfig:
    """One validation-only transformer candidate."""

    model: str
    learning_rate: float = 2e-5
    max_length: int = 128
    weight_decay: float = 0.01

    @property
    def key(self) -> str:
        model = self.model.rsplit("/", maxsplit=1)[-1]
        return f"{model}|lr={self.learning_rate:g}|len={self.max_length}|wd={self.weight_decay:g}"


def encoder_candidates() -> list[StudyConfig]:
    """Return the preregistered equal-footing encoder screen."""

    return [StudyConfig(model=model) for model in ENCODERS]


def tuning_candidates(model: str) -> list[StudyConfig]:
    """Return a bounded coordinate search around the common baseline."""

    return [
        StudyConfig(model=model),
        StudyConfig(model=model, learning_rate=1e-5),
        StudyConfig(model=model, learning_rate=3e-5),
        StudyConfig(model=model, max_length=96),
        StudyConfig(model=model, max_length=192),
        StudyConfig(model=model, weight_decay=0.0),
    ]


def select_validation_winner(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select only by inner-validation macro-F1 with a stable tie break."""

    if not records:
        raise ValueError("At least one validation result is required")
    return max(records, key=lambda item: (item["validation_macro_f1"], item["config"]["key"]))


def _summary(folds: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    values = [float(fold[metric]) for fold in folds]
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
    }


def merge_study_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge independently computed outer folds after strict provenance checks."""

    if not reports:
        raise ValueError("At least one study report is required")
    reference = reports[0]
    protocol_keys = (
        "outer_folds",
        "validation_splits",
        "epochs_ceiling",
        "batch_size",
        "early_stopping_patience",
        "selection_metric",
        "selection_seed",
        "stability_seeds",
        "encoder_candidates",
        "learning_rates",
        "max_lengths",
        "weight_decays",
        "class_weighting",
        "seed_policy",
    )
    for report in reports[1:]:
        if report.get("dataset") != reference.get("dataset"):
            raise ValueError("Cannot merge reports produced from different datasets")
        if any(
            report["protocol"].get(key) != reference["protocol"].get(key) for key in protocol_keys
        ):
            raise ValueError("Cannot merge reports produced with different protocols")

    folds = sorted(
        (fold for report in reports for fold in report["folds"]), key=lambda item: item["fold"]
    )
    fold_numbers = [fold["fold"] for fold in folds]
    if len(fold_numbers) != len(set(fold_numbers)):
        raise ValueError("Cannot merge duplicate outer folds")
    expected = list(range(1, int(reference["protocol"]["outer_folds"]) + 1))
    if fold_numbers != expected:
        raise ValueError(f"Complete evidence requires outer folds {expected}, got {fold_numbers}")

    selected_models = Counter(fold["selected_config"]["model"] for fold in folds)
    merged = {key: value for key, value in reference.items() if key not in {"folds", "summary"}}
    merged["protocol"] = {**reference["protocol"], "completed_outer_folds": expected}
    merged["folds"] = folds
    merged["environments_by_fold"] = {
        str(report["folds"][0]["fold"]): report["environment"] for report in reports
    }
    merged["summary"] = {
        "accuracy": _summary(folds, "accuracy"),
        "macro_f1": _summary(folds, "macro_f1"),
        "selected_model_counts": dict(sorted(selected_models.items())),
    }
    merged["created_at_utc"] = datetime.now(UTC).isoformat()
    return merged


def run_transformer_study(
    frame: Any,
    *,
    outer_fold_numbers: Sequence[int],
    n_splits: int = 5,
    epochs: float = 4.0,
    batch_size: int = 32,
    random_state: int = 42,
    validation_splits: int = 5,
    early_stopping_patience: int = 1,
    stability_seeds: Sequence[int] = STABILITY_SEEDS,
    expected_test_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run nested encoder/config selection without consulting outer labels."""

    requested = sorted(set(outer_fold_numbers))
    if not requested or requested[0] < 1 or requested[-1] > n_splits:
        raise ValueError(f"outer folds must be between 1 and {n_splits}")
    if not stability_seeds or stability_seeds[0] != random_state:
        raise ValueError("stability seeds must start with the canonical random state")
    outer_folds = transformer_folds(frame, n_splits=n_splits, random_state=random_state)
    actual_test_rows = [len(test) for _, test in outer_folds]
    if expected_test_rows is not None and actual_test_rows != list(expected_test_rows):
        raise RuntimeError("Transformer folds do not match the reference benchmark")

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("Install the transformer extra before running the GPU study") from error

    class BalancedTrainer(Trainer):
        def __init__(self, *args: Any, label_weights: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.label_weights = label_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=self.label_weights.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    texts = transformer_inputs(frame)
    labels = frame["sentiment"].map(LABEL_TO_ID).tolist()
    completed: list[dict[str, Any]] = []

    def train_candidate(
        config: StudyConfig,
        fit_indices: Any,
        validation_indices: Any,
        *,
        seed: int,
        output_dir: Path,
    ) -> tuple[str, Any, dict[str, Any]]:
        set_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )
        tokenize = partial(_tokenize_batch, tokenizer=tokenizer, max_length=config.max_length)

        def dataset(indices: Any) -> Any:
            source = Dataset.from_dict(
                {
                    "input_text": [texts[index] for index in indices],
                    "label": [labels[index] for index in indices],
                }
            )
            return source.map(tokenize, batched=True, remove_columns=["input_text"])

        train_data = dataset(fit_indices)
        validation_data = dataset(validation_indices)
        weight_map = balanced_label_weights([ID_TO_LABEL[labels[index]] for index in fit_indices])
        weights = torch.tensor([weight_map[label] for label in LABELS], dtype=torch.float32)

        def validation_metrics(output: Any) -> dict[str, float]:
            predicted = np.asarray(output.predictions).argmax(axis=1)
            actual = [ID_TO_LABEL[index] for index in output.label_ids]
            inferred = [ID_TO_LABEL[index] for index in predicted]
            metrics = classification_metrics(actual, inferred)
            return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

        arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
            seed=seed,
            data_seed=seed,
            fp16=torch.cuda.is_available(),
        )
        trainer = BalancedTrainer(
            model=model,
            args=arguments,
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            processing_class=tokenizer,
            compute_metrics=validation_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
            label_weights=weights,
        )
        started = perf_counter()
        trainer.train()
        checkpoint = trainer.state.best_model_checkpoint
        if checkpoint is None:
            raise RuntimeError("Training completed without a selected checkpoint")
        record = {
            "config": {**asdict(config), "key": config.key},
            "seed": seed,
            "validation_macro_f1": float(trainer.state.best_metric),
            "epochs_trained": trainer.state.epoch,
            "fit_seconds": perf_counter() - started,
            "label_weights": weight_map,
        }
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return checkpoint, tokenizer, record

    for fold_number in requested:
        train_indices, test_indices = outer_folds[fold_number - 1]
        fit_indices, validation_indices = grouped_validation_split(
            frame,
            train_indices,
            n_splits=validation_splits,
            random_state=random_state,
        )
        print(f"Starting nested transformer study fold {fold_number}/{n_splits}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"sentiment-study-{fold_number}-") as temp:
            root = Path(temp)
            screened: dict[str, tuple[str, Any, dict[str, Any]]] = {}
            for config in encoder_candidates():
                result = train_candidate(
                    config,
                    fit_indices,
                    validation_indices,
                    seed=random_state,
                    output_dir=root / f"screen-{len(screened)}",
                )
                screened[config.key] = result
            encoder_records = [item[2] for item in screened.values()]
            encoder_winner = select_validation_winner(encoder_records)
            selected_model = encoder_winner["config"]["model"]

            tuned = dict(screened)
            for config in tuning_candidates(selected_model):
                if config.key in tuned:
                    continue
                result = train_candidate(
                    config,
                    fit_indices,
                    validation_indices,
                    seed=random_state,
                    output_dir=root / f"tune-{len(tuned)}",
                )
                tuned[config.key] = result
            tuning_records = [
                item[2]
                for key, item in tuned.items()
                if item[2]["config"]["model"] == selected_model
            ]
            winner = select_validation_winner(tuning_records)
            selected = StudyConfig(
                model=winner["config"]["model"],
                learning_rate=winner["config"]["learning_rate"],
                max_length=winner["config"]["max_length"],
                weight_decay=winner["config"]["weight_decay"],
            )

            seed_results: list[dict[str, Any]] = []
            primary_metrics: dict[str, Any] | None = None
            artifact_bytes = 0
            for seed in stability_seeds:
                if seed == random_state:
                    checkpoint, tokenizer, training = tuned[selected.key]
                else:
                    checkpoint, tokenizer, training = train_candidate(
                        selected,
                        fit_indices,
                        validation_indices,
                        seed=seed,
                        output_dir=root / f"seed-{seed}",
                    )
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                predictor = Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=root / f"predict-{seed}",
                        per_device_eval_batch_size=batch_size * 2,
                        report_to="none",
                    ),
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    processing_class=tokenizer,
                )
                tokenize = partial(
                    _tokenize_batch,
                    tokenizer=tokenizer,
                    max_length=selected.max_length,
                )
                test_data = Dataset.from_dict(
                    {
                        "input_text": [texts[index] for index in test_indices],
                        "label": [labels[index] for index in test_indices],
                    }
                ).map(tokenize, batched=True, remove_columns=["input_text"])
                started = perf_counter()
                output = predictor.predict(test_data)
                predict_seconds = perf_counter() - started
                predicted = np.asarray(output.predictions).argmax(axis=1)
                actual = [ID_TO_LABEL[index] for index in output.label_ids]
                inferred = [ID_TO_LABEL[index] for index in predicted]
                metrics = classification_metrics(actual, inferred)
                seed_result = {
                    "seed": seed,
                    "validation_macro_f1": training["validation_macro_f1"],
                    "epochs_trained": training["epochs_trained"],
                    "fit_seconds": training["fit_seconds"],
                    "predict_seconds": predict_seconds,
                    **metrics,
                }
                seed_results.append(seed_result)
                if seed == random_state:
                    primary_metrics = metrics
                    artifact = root / "selected-artifact"
                    model.save_pretrained(artifact)
                    tokenizer.save_pretrained(artifact)
                    artifact_bytes = _artifact_size(artifact)
                del predictor, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            assert primary_metrics is not None
            diagnostics = split_diagnostics(frame.iloc[train_indices], frame.iloc[test_indices])
            completed.append(
                {
                    "fold": fold_number,
                    **diagnostics,
                    "fit_rows": len(fit_indices),
                    "validation_rows": len(validation_indices),
                    "encoder_screen": encoder_records,
                    "tuning_search": tuning_records,
                    "selected_config": {**asdict(selected), "key": selected.key},
                    "selection_metric": winner["validation_macro_f1"],
                    "seed_results": seed_results,
                    "artifact_bytes": artifact_bytes,
                    **primary_metrics,
                }
            )

    selected_models = Counter(fold["selected_config"]["model"] for fold in completed)
    return {
        "protocol": {
            "strategy": "nested grouped encoder screen and coordinate hyperparameter search",
            "outer_folds": n_splits,
            "completed_outer_folds": requested,
            "validation_splits": validation_splits,
            "epochs_ceiling": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
            "selection_metric": "inner_validation_macro_f1",
            "selection_seed": random_state,
            "stability_seeds": list(stability_seeds),
            "class_weighting": "balanced",
            "encoder_candidates": list(ENCODERS),
            "learning_rates": [1e-5, 2e-5, 3e-5],
            "max_lengths": [96, 128, 192],
            "weight_decays": [0.0, 0.01],
            "seed_policy": "measure stability; never select a seed by outer-test performance",
        },
        "folds": completed,
        "summary": {
            "accuracy": _summary(completed, "accuracy"),
            "macro_f1": _summary(completed, "macro_f1"),
            "selected_model_counts": dict(sorted(selected_models.items())),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "accelerate": version("accelerate"),
            "scikit_learn": version("scikit-learn"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }


def transformer_study_command(argv: Sequence[str] | None = None) -> None:
    """Run one or more independently recoverable outer study folds."""

    parser = argparse.ArgumentParser(description="Nested grouped transformer study.")
    parser.add_argument("--data", type=Path, default=Path("data/twitter_training.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, action="append", dest="outer_folds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-seeds", type=int, nargs="+", default=list(STABILITY_SEEDS))
    args = parser.parse_args(argv)
    frame, profile = load_dataset(args.data)
    expected = _reference_test_rows(
        args.reference,
        dataset_sha256=profile.sha256,
        n_splits=args.folds,
    )
    report = run_transformer_study(
        frame,
        outer_fold_numbers=args.outer_folds or list(range(1, args.folds + 1)),
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
        validation_splits=args.validation_folds,
        early_stopping_patience=args.early_stopping_patience,
        stability_seeds=args.stability_seeds,
        expected_test_rows=expected,
    )
    report.update({"created_at_utc": datetime.now(UTC).isoformat(), "dataset": profile.to_dict()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


def transformer_study_merge_command(argv: Sequence[str] | None = None) -> None:
    """Merge five independently downloaded study-fold reports."""

    parser = argparse.ArgumentParser(description="Merge nested transformer study folds.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    merged = merge_study_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": merged["summary"]}, indent=2))
