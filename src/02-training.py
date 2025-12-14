"""
Transformer (HuBERT) trainer for the legal text decoder task.

Inputs:
    data/final/train.csv
    data/final/test.csv
Each CSV: text,label

Outputs:
    /app/model.pt
    /app/model_metrics.json
Logs:
    stdout (redirect to log/run.log in Docker)
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import config
from utils import setup_logger

logger = setup_logger(__name__, "log/training.log")


class ASZFDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: Series, labels: Series):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=config.HF_MAX_LENGTH,
        )
        self.labels = labels.astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx] - 1, dtype=torch.long),  # labels -> 0..4
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log_hyperparams() -> None:
    logger.info(
        "Hyperparameters -> model: %s | epochs: %s | batch: %s | lr: %s | max_len: %s | warmup: %s | weight_decay: %s | grad_clip: %s",
        config.HF_MODEL_NAME,
        config.HF_EPOCHS,
        config.HF_BATCH_SIZE,
        config.HF_LR,
        config.HF_MAX_LENGTH,
        config.HF_WARMUP_STEPS,
        config.HF_WEIGHT_DECAY,
        config.HF_GRAD_CLIP,
    )


def _load_splits(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train CSV at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test CSV at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for name, df in (("train", train_df), ("test", test_df)):
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"{name} CSV must contain columns: text,label")
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

    logger.info("Loaded data. Train rows: %d, Test rows: %d", len(train_df), len(test_df))
    return train_df, test_df


def _log_label_distribution(name: str, labels: Series) -> None:
    counts = labels.value_counts().sort_index()
    dist = ", ".join(f"{cls}:{cnt}" for cls, cnt in counts.items())
    logger.info("%s label distribution -> %s", name, dist)


def _get_dataloaders(tokenizer: AutoTokenizer, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_part, val_part = train_test_split(
        train_df,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=train_df["label"],
    )
    _log_label_distribution("Train (fit)", train_part["label"])
    _log_label_distribution("Validation", val_part["label"])
    if len(test_df):
        _log_label_distribution("Test", test_df["label"])

    train_ds = ASZFDataset(tokenizer, train_part["text"], train_part["label"])
    val_ds = ASZFDataset(tokenizer, val_part["text"], val_part["label"])
    test_ds = ASZFDataset(tokenizer, test_df["text"], test_df["label"])

    train_loader = DataLoader(train_ds, batch_size=config.HF_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.HF_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.HF_BATCH_SIZE)
    return train_loader, val_loader, test_loader


def _eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def train() -> None:
    _seed_everything(config.RANDOM_STATE)
    device = _device()
    logger.info("Using device: %s", device)
    _log_hyperparams()

    train_df, test_df = _load_splits(Path(config.TRAIN_CSV), Path(config.TEST_CSV))
    _log_label_distribution("Train", train_df["label"])

    tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.HF_MODEL_NAME,
        num_labels=5,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: total=%d trainable=%d", total_params, trainable_params)

    train_loader, val_loader, test_loader = _get_dataloaders(tokenizer, train_df, test_df)

    total_steps = len(train_loader) * config.HF_EPOCHS
    optimizer = AdamW(
        model.parameters(),
        lr=config.HF_LR,
        weight_decay=config.HF_WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.HF_WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_f1 = -math.inf
    best_state = None
    history = {"loss": [], "val_f1": []}

    for epoch in range(1, config.HF_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.HF_GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_time = time.perf_counter() - t0
        avg_loss = epoch_loss / max(1, len(train_loader))
        val_metrics = _eval(model, val_loader, device)
        logger.info(
            "Epoch %d/%d | loss: %.4f | val_acc: %.4f | val_f1_macro: %.4f | time: %.2fs",
            epoch,
            config.HF_EPOCHS,
            avg_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
            epoch_time,
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = model.state_dict()

        history["loss"].append(avg_loss)
        history["val_f1"].append(val_metrics["f1_macro"])

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = _eval(model, test_loader, device) if len(test_df) else {"accuracy": None, "f1_macro": None}
    logger.info("Test metrics: %s", test_metrics)

    model_path = Path(config.HF_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model weights to %s", model_path)

    metrics = {
        "val_best_f1_macro": best_val_f1,
        "test": test_metrics,
    }
    metrics_path = model_path.with_suffix(".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # Save training curves
    media_dir = Path("media")
    media_dir.mkdir(parents=True, exist_ok=True)
    if history["loss"]:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(history["loss"]) + 1), history["loss"], label="Train loss")
        plt.plot(range(1, len(history["val_f1"]) + 1), history["val_f1"], label="Val F1")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Training curves")
        plt.tight_layout()
        curve_path = media_dir / "training_curves.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        logger.info("Saved training curves to %s", curve_path)


if __name__ == "__main__":
    train()
