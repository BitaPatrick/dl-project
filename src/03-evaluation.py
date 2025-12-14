"""
Evaluation script for the legal text decoder (HuBERT classifier).

Loads:
    - data/final/test.csv
    - /app/model.pt  (weights saved by 02-training.py)
Runs inference and logs accuracy + macro F1 on the test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas import Series
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from utils import setup_logger

logger = setup_logger(__name__, "log/eval.log")


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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_test(test_path: Path) -> pd.DataFrame:
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test CSV at {test_path}")
    df = pd.read_csv(test_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: text,label")
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def _resolve_model_path() -> Path:
    primary = Path(config.HF_MODEL_PATH)
    fallback = Path("/app/model.pt")
    if primary.exists():
        return primary
    if fallback.exists():
        logger.warning("Primary model path missing (%s); using fallback %s", primary, fallback)
        return fallback
    raise FileNotFoundError(f"Model weights not found at {primary} (or fallback {fallback}). Run 02-training.py after rebuilding the image.")


def _eval(model, loader, device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu())
            labels.append(lbls.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, digits=3)
    logger.info("Classification report:\n%s", report)
    return {"accuracy": acc, "f1_macro": f1}, preds.numpy(), labels.numpy()


def _save_confusion(labels: np.ndarray, preds: np.ndarray, split_name: str, out_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({split_name})")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_{split_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix for %s to %s", split_name, out_path)


def evaluate() -> None:
    device = _device()
    logger.info("Using device: %s", device)

    # load splits
    train_df = pd.read_csv(Path(config.TRAIN_CSV))
    test_df = _load_test(Path(config.TEST_CSV))
    train_df["text"] = train_df["text"].astype(str).str.strip()
    train_df["label"] = train_df["label"].astype(int)
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # build validation split from train to mirror training
    if len(train_df) == 0:
        val_df = pd.DataFrame()
    else:
        train_df, val_df = train_test_split(
            train_df,
            test_size=config.VAL_SPLIT,
            random_state=config.RANDOM_STATE,
            stratify=train_df["label"],
        )
        logger.info("Validation rows: %d", len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.HF_MODEL_NAME,
        num_labels=5,
    )

    model_weights = _resolve_model_path()
    state = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    metrics: Dict[str, Dict[str, float]] = {}
    media_dir = Path("media")

    if len(train_df) and 'val_df' in locals() and len(val_df):
        val_ds = ASZFDataset(tokenizer, val_df["text"], val_df["label"])
        val_loader = DataLoader(val_ds, batch_size=config.HF_BATCH_SIZE)
        logger.info("Running validation evaluation...")
        val_metrics, val_preds, val_labels = _eval(model, val_loader, device)
        metrics["val"] = val_metrics
        _save_confusion(val_labels, val_preds, "val", media_dir)

    test_ds = ASZFDataset(tokenizer, test_df["text"], test_df["label"])
    test_loader = DataLoader(test_ds, batch_size=config.HF_BATCH_SIZE)
    logger.info("Running test evaluation...")
    test_metrics, test_preds, test_labels = _eval(model, test_loader, device)
    metrics["test"] = test_metrics
    _save_confusion(test_labels, test_preds, "test", media_dir)

    if "val" in metrics:
        logger.info("Validation metrics -> acc: %.4f | f1_macro: %.4f", metrics["val"]["accuracy"], metrics["val"]["f1_macro"])
    logger.info("Test metrics -> acc: %.4f | f1_macro: %.4f", metrics["test"]["accuracy"], metrics["test"]["f1_macro"])

    metrics_path = model_weights.with_suffix(".eval.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)


if __name__ == "__main__":
    evaluate()
