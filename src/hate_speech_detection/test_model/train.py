"""
train.py — Fine-tuning Transformer models for Vietnamese Hate Speech Detection (HSD)
=====================================================================================
Dataset  : data_final_sorted_cleaned_hsd.csv  (21 497 reviews, ITViec)
Labels   : CLEAN | OFFENSIVE | HATE  (3-class sequence classification)
Default  : vinai/phobert-large

Usage examples
--------------
# Full GPU run (NVIDIA L40S, 48 GB VRAM)
python train.py --gpu_mode

# Swap model
python train.py --gpu_mode --model_name "vinai/phobert-base-v2"

# Sanity / smoke test on CPU (1 000 samples, 1 epoch, batch 2)
python train.py --smoke_test

# Disable wandb logging
python train.py --gpu_mode --no_wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL2ID: Dict[str, int] = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

TEXT_COLUMN   = "review_content_cleaned"
LABEL_COLUMN  = "category"
RANDOM_SEED   = 42

DATA_PATH = Path(__file__).parent / "data_final_sorted_cleaned_hsd.csv"
HISTORY_PATH = Path(__file__).parent / "experiment_history.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace Transformer for Vietnamese HSD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/phobert-large",
        help="HuggingFace model hub ID or local path. "
             "Tokenizer is loaded from the same identifier.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DATA_PATH,
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Root output directory. Defaults to ./saved_models/<model-slug>-hsd",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a quick smoke test: CPU, 1 000 samples, 1 epoch, batch size 2.",
    )
    parser.add_argument(
        "--gpu_mode",
        action="store_true",
        help="Optimise for NVIDIA L40S (48 GB VRAM): bf16, tf32, large batch.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum tokenisation length (tokens).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs (ignored in smoke_test mode).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for LR warm-up.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW regularisation.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (evaluation steps). Set 0 to disable.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="hsd-itviec",
        help="W&B project name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_and_split(
    data_path: Path,
    smoke_test: bool,
    seed: int,
) -> pd.DataFrame:
    """Load CSV and return train / val / test DataFrames via stratified split."""
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    # Drop rows with missing text or label
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.upper()
    df = df[df[LABEL_COLUMN].isin(LABEL2ID.keys())].reset_index(drop=True)

    logger.info(
        "Dataset size after cleaning: %d  |  Distribution: %s",
        len(df),
        df[LABEL_COLUMN].value_counts().to_dict(),
    )

    if smoke_test:
        df = df.sample(n=min(1_000, len(df)), random_state=seed).reset_index(drop=True)
        logger.warning("SMOKE TEST: using %d samples.", len(df))

    # Stratified 80 / 10 / 10
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df[LABEL_COLUMN], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[LABEL_COLUMN], random_state=seed
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def df_to_hf_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DatasetDict:
    """Convert pandas DataFrames to a HuggingFace DatasetDict."""

    def _convert(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": df[TEXT_COLUMN].tolist(),
                "label": [LABEL2ID[lbl] for lbl in df[LABEL_COLUMN]],
            }
        )

    return DatasetDict(
        {
            "train": _convert(train_df),
            "validation": _convert(val_df),
            "test": _convert(test_df),
        }
    )


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def build_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load a tokenizer from the HuggingFace hub (or local path).

    NOTE — word segmentation:
      PhoBERT models (vinai/phobert-*) were pre-trained on word-segmented
      Vietnamese text produced by VnCoreNLP. For maximum accuracy you should
      pre-segment `review_content_cleaned` with `py_vncorenlp` before
      tokenisation:

          from py_vncorenlp import VnCoreNLP
          rdrsegmenter = VnCoreNLP(annotators=["wseg"])
          text = " ".join(rdrsegmenter.word_segment(text))

      This script skips segmentation for simplicity / portability.
      Models that use a BPE/SentencePiece tokenizer (e.g. xlm-roberta-*)
      do NOT require pre-segmentation.
    """
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> DatasetDict:
    """Batch-tokenise all splits."""
    logger.info("Tokenising dataset (max_length=%d)…", max_length)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,          # dynamic padding via DataCollatorWithPadding
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    tokenized.set_format("torch")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics():
    """Return a closure that computes Macro-F1 + accuracy."""
    f1_metric  = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1  = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        acc = acc_metric.compute(predictions=predictions, references=labels)
        return {"f1_macro": f1["f1"], "accuracy": acc["accuracy"]}

    return compute_metrics


# ---------------------------------------------------------------------------
# Training arguments factory
# ---------------------------------------------------------------------------

def build_training_args(
    output_dir: Path,
    smoke_test: bool,
    gpu_mode: bool,
    num_train_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    report_to: List[str],
) -> TrainingArguments:
    """Construct TrainingArguments for the three supported modes."""

    # ---- smoke test (CPU, tiny) ----
    if smoke_test:
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=False,
            bf16=False,
            use_cpu=True,           # replaces deprecated no_cuda (transformers >= 4.37)
            report_to=report_to,
            seed=RANDOM_SEED,
        )

    # ---- NVIDIA L40S GPU (48 GB VRAM) ----
    if gpu_mode:
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            bf16=True,          # bfloat16 — native on L40S / Ada Lovelace
            tf32=True,          # TensorFloat-32 matmul — free throughput gain
            fp16=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to=report_to,
            seed=RANDOM_SEED,
        )

    # ---- Default CPU / single-GPU ----
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        report_to=report_to,
        seed=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Experiment history (local leaderboard)
# ---------------------------------------------------------------------------

def update_experiment_history(
    history_path: Path,
    model_name: str,
    f1_macro: float,
    eval_loss: float,
    output_dir: Path,
    extra: Optional[Dict] = None,
) -> None:
    """
    Append one run's result to experiment_history.json.

    Schema per entry:
    {
        "model_name":  "vinai/phobert-large",
        "timestamp":   "2025-06-01T12:34:56Z",
        "f1_macro":    0.8734,
        "eval_loss":   0.3142,
        "saved_model": "./saved_models/phobert-large-hsd",
        ...extra fields...
    }
    """
    history: List[Dict] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except json.JSONDecodeError:
            logger.warning("experiment_history.json is malformed — starting fresh.")

    entry = {
        "model_name":  model_name,
        "timestamp":   datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "f1_macro":    round(f1_macro, 6),
        "eval_loss":   round(eval_loss, 6),
        "saved_model": str(output_dir),
        **(extra or {}),
    }
    history.append(entry)

    # Sort descending by f1_macro so the best run is always at the top
    history.sort(key=lambda r: r.get("f1_macro", 0.0), reverse=True)

    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    logger.info("Experiment history updated → %s", history_path)
    logger.info("Leaderboard (top-5):")
    for rank, run in enumerate(history[:5], 1):
        logger.info(
            "  #%d  %-45s  F1=%.4f  Loss=%.4f  [%s]",
            rank, run["model_name"], run["f1_macro"], run["eval_loss"], run["timestamp"],
        )


# ---------------------------------------------------------------------------
# Model slug helper
# ---------------------------------------------------------------------------

def model_slug(model_name: str) -> str:
    """Convert 'vinai/phobert-large' → 'phobert-large'."""
    slug = model_name.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "-", slug)
    return slug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ---- wandb setup ----
    report_to: List[str] = []
    if not args.no_wandb:
        try:
            import wandb  # noqa: F401

            run_name = f"{model_slug(args.model_name)}-{'smoke' if args.smoke_test else 'full'}"
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            report_to = ["wandb"]
            logger.info("W&B enabled — project: %s | run: %s", args.wandb_project, run_name)
        except ImportError:
            logger.warning("wandb not installed — disabling W&B logging. Run: pip install wandb")

    # ---- output directory ----
    slug = model_slug(args.model_name)
    output_dir: Path = args.output_dir or (
        Path(__file__).parent / "saved_models" / f"{slug}-hsd"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ---- data ----
    train_df, val_df, test_df = load_and_split(
        args.data_path, smoke_test=args.smoke_test, seed=args.seed
    )
    raw_datasets = df_to_hf_dataset(train_df, val_df, test_df)

    # ---- tokenizer + tokenisation ----
    tokenizer = build_tokenizer(args.model_name)
    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, max_length=args.max_length)

    # ---- model ----
    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # safe when swapping classifier head
    )

    # ---- training args ----
    training_args = build_training_args(
        output_dir=output_dir,
        smoke_test=args.smoke_test,
        gpu_mode=args.gpu_mode,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        report_to=report_to,
    )

    # ---- callbacks ----
    callbacks = []
    if args.early_stopping_patience > 0 and not args.smoke_test:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    # ---- trainer ----
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,   # replaces deprecated 'tokenizer' arg (transformers >= 4.46)
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
    )

    # ---- train ----
    logger.info("=" * 60)
    logger.info("Starting training — model: %s", args.model_name)
    logger.info("  smoke_test=%s | gpu_mode=%s | epochs=%d",
                args.smoke_test, args.gpu_mode, training_args.num_train_epochs)
    logger.info("=" * 60)

    train_result = trainer.train()

    # ---- save final model ----
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model + tokenizer saved to %s", output_dir)

    # ---- evaluate on held-out test set ----
    logger.info("Evaluating on test set…")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    logger.info("Test metrics: %s", test_metrics)

    # ---- evaluate on validation set (for history logging) ----
    val_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        metric_key_prefix="eval",
    )

    f1_macro  = val_metrics.get("eval_f1_macro",  test_metrics.get("test_f1_macro",  0.0))
    eval_loss = val_metrics.get("eval_loss",       test_metrics.get("test_loss",      0.0))

    # ---- local experiment registry ----
    update_experiment_history(
        history_path=HISTORY_PATH,
        model_name=args.model_name,
        f1_macro=f1_macro,
        eval_loss=eval_loss,
        output_dir=output_dir,
        extra={
            "test_f1_macro":  test_metrics.get("test_f1_macro"),
            "test_loss":      test_metrics.get("test_loss"),
            "train_runtime":  train_result.metrics.get("train_runtime"),
            "train_samples":  len(tokenized_datasets["train"]),
            "max_length":     args.max_length,
            "epochs":         training_args.num_train_epochs,
            "batch_size":     training_args.per_device_train_batch_size,
            "learning_rate":  args.learning_rate,
            "smoke_test":     args.smoke_test,
            "gpu_mode":       args.gpu_mode,
            "seed":           args.seed,
        },
    )

    logger.info("Done. Best model checkpoint: %s", output_dir)


if __name__ == "__main__":
    main()
