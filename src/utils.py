import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__,
    file_path: Optional[str] = None,
    master_file: str = "log/run.log",
) -> logging.Logger:
    """
    Sets up a logger that outputs to stdout, a master run log, and optionally a per-script log.
    Avoids adding duplicate handlers on repeated calls.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # master run log
    if master_file:
        master_path = Path(master_file)
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_handler = logging.FileHandler(master_path, mode="a", encoding="utf-8")
        master_handler.setFormatter(formatter)
        logger.addHandler(master_handler)

    # optional per-script log
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config():
    pass
