"""
Baseline (classical ML) trainer for the legal text decoder task.

Pipeline:
- TF-IDF (uni+bi-grams)
- L2 Logistic Regression with class weights

Inputs:
    data/final/train.csv
    data/final/test.csv

Outputs:
    /app/model.joblib
    /app/model.metrics.json
Logs:
    stdout (redirect to log/run.log in Docker)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import config
from utils import setup_logger

logger = setup_logger(__name__, "log/baseline.log")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def _build_pipeline(class_weights: Dict[int, float]) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(
        C=config.LOG_REG_C,
        max_iter=config.LOG_REG_MAX_ITER,
        class_weight=class_weights,
        n_jobs=-1,
        multi_class="auto",
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )


def _compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    unique_labels = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels)
    return {cls: w for cls, w in zip(unique_labels, weights)}


def _log_label_distribution(name: str, labels: Series) -> None:
    counts = labels.value_counts().sort_index()
    dist = ", ".join(f"{cls}:{cnt}" for cls, cnt in counts.items())
    logger.info("%s label distribution -> %s", name, dist)


def _eval_split(name: str, model: Pipeline, X: pd.Series, y: pd.Series) -> Dict[str, float]:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average="macro")
    f1_weighted = f1_score(y, preds, average="weighted")
    logger.info("%s accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f", name, acc, f1_macro, f1_weighted)
    report = classification_report(y, preds, digits=3)
    logger.info("%s classification report:\n%s", name, report)
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def train() -> None:
    _seed_everything(config.RANDOM_STATE)

    train_df, test_df = _load_splits(Path(config.TRAIN_CSV), Path(config.TEST_CSV))
    _log_label_distribution("Train", train_df["label"])
    if len(test_df):
        _log_label_distribution("Test", test_df["label"])

    train_part, val_part = train_test_split(
        train_df,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=train_df["label"],
    )
    _log_label_distribution("Train (fit)", train_part["label"])
    _log_label_distribution("Validation", val_part["label"])

    class_weights = _compute_class_weights(train_part["label"].values)
    logger.info("Class weights: %s", class_weights)

    model = _build_pipeline(class_weights)
    logger.info("Fitting model (features=%d, ngram=%s)", config.MAX_FEATURES, config.NGRAM_RANGE)
    t0 = time.perf_counter()
    model.fit(train_part["text"], train_part["label"])
    logger.info("Fit finished in %.2fs", time.perf_counter() - t0)

    metrics = {
        "val": _eval_split("Validation", model, val_part["text"], val_part["label"]),
    }

    if len(test_df):
        metrics["test"] = _eval_split("Test", model, test_df["text"], test_df["label"])
    else:
        logger.warning("Test split is empty. Skipping test evaluation.")

    model_path = Path(config.MODEL_SAVE_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_path = model_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    train()
