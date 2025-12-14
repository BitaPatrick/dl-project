"""Data preprocessing script.

Steps:
- Download the dataset zip from config.DATA_URL into DATA_DIR.
- Extract it under DATA_DIR/raw.
- Load JSON files from each Neptun directory (skip `sample` and `consensus`).
- Aggregate records and create fixed train/test splits based on Neptun codes.
"""
from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

import config
from utils import setup_logger

logger = setup_logger(__name__, "log/preprocess.log")

EXCLUDE_DIRS = {"sample", "consensus"}
MIN_TEXT_LENGTH = 15


def _ensure_data_url(url: str) -> str:
    if not url:
        raise ValueError("config.DATA_URL is empty. Please set the dataset zip URL in config.py.")
    return url


def download_zip(url: str, dest: Path) -> Path:
    """Download the dataset zip to dest if it does not already exist."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Zip already exists at %s, skipping download.", dest)
        return dest

    logger.info("Downloading dataset from %s", url)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    size_mb = dest.stat().st_size / 1e6
    logger.info("Download complete: %s (%.2f MB)", dest, size_mb)
    return dest


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract zip_path into target_dir/raw (clean it first)."""
    extract_root = target_dir / "raw"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s to %s", zip_path, extract_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return extract_root


def _iter_neptun_dirs(root: Path) -> Iterable[Tuple[str, Path]]:
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        if name.lower() in EXCLUDE_DIRS:
            logger.info("Skipping excluded directory: %s", name)
            continue
        yield name, item


def _load_json_files(neptun_dir: Path) -> List[Dict]:
    records: List[Dict] = []
    for json_file in neptun_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read %s (%s); skipping.", json_file, exc)
            continue

        # Files can contain a single dict or a list of Label Studio tasks
        tasks = payload if isinstance(payload, list) else [payload]
        total_in_file = len(tasks)
        skipped_in_file = 0
        for task in tasks:
            if not isinstance(task, dict):
                skipped_in_file += 1
                continue

            text = task.get("text") or task.get("data", {}).get("text")
            label = task.get("label")

            # Extract label from annotation choices if not present at top level
            if label is None:
                annotations = task.get("annotations") or []
                for annotation in annotations:
                    if annotation.get("was_cancelled"):
                        continue
                    results = annotation.get("result") or []
                    for result in results:
                        choices = result.get("value", {}).get("choices") or []
                        if choices:
                            label = choices[0]
                            break
                    if label is not None:
                        break

            if text is None or label is None:
                skipped_in_file += 1
                continue

            text = text.strip()
            if not text or len(text) < MIN_TEXT_LENGTH:
                skipped_in_file += 1
                continue

            # Normalise label strings like '3-Többé/kevésbé megértem' to the numeric prefix if available
            if isinstance(label, str):
                prefix = label.split("-", 1)[0].strip()
                if prefix.isdigit():
                    label = int(prefix)

            records.append({"text": text, "label": label})
        if skipped_in_file:
            logger.warning(
                "Missing text/label in %s; skipped %d of %d entries.",
                json_file,
                skipped_in_file,
                total_in_file,
            )
    return records


def aggregate_records(root: Path) -> Dict[str, List[Dict]]:
    """Aggregate JSON records per Neptun code."""
    data: Dict[str, List[Dict]] = {}
    for neptun, folder in _iter_neptun_dirs(root):
        records = _load_json_files(folder)
        if not records:
            logger.info("No JSON records found for %s; skipping.", neptun)
            continue
        data[neptun] = records
        logger.info("Loaded %d records for %s", len(records), neptun)
    return data


def split_train_test(
    data_by_neptun: Dict[str, List[Dict]],
    test_codes: Iterable[str],
) -> Tuple[List[Dict], List[Dict]]:
    """Split records into train/test by Neptun codes."""
    test_codes_set = {code.upper() for code in test_codes}
    train, test = [], []

    for neptun, records in data_by_neptun.items():
        target_list = test if neptun.upper() in test_codes_set else train
        # Keep track of origin Neptun for traceability
        for rec in records:
            rec_with_origin = {**rec, "neptun": neptun}
            target_list.append(rec_with_origin)

    logger.info("Split complete. Train: %d records, Test: %d records", len(train), len(test))
    return train, test


def preprocess():
    """Full preprocessing pipeline returning (train, test) records."""
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = _ensure_data_url(config.DATA_URL)
    zip_name = Path(url).name or "dataset.zip"
    zip_path = data_dir / zip_name

    downloaded = download_zip(url, zip_path)
    extracted_root = extract_zip(downloaded, data_dir)

    # Handle zip structures that wrap contents in a single folder (e.g., legaltextdecoder/)
    possible_roots = [p for p in extracted_root.iterdir() if p.is_dir()]
    dataset_root = extracted_root
    if len(possible_roots) == 1 and not any(extracted_root.glob("*.json")):
        dataset_root = possible_roots[0]
        logger.info("Detected nested dataset folder: %s", dataset_root.name)

    aggregated = aggregate_records(dataset_root)
    train, test = split_train_test(aggregated, config.TEST_NEPTUN_CODES)

    # Persist splits for downstream steps
    final_dir = data_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(records, path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for rec in records:
                writer.writerow([rec["text"], rec["label"]])

    _write_split(train, final_dir / "train.csv")
    _write_split(test, final_dir / "test.csv")

    logger.info("Preprocessing finished. Total records: %d", len(train) + len(test))
    return train, test


if __name__ == "__main__":
    preprocess()
