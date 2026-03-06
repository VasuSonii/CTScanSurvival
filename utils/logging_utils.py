"""
utils/logging_utils.py
=======================
Shared logging setup used by both training scripts.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_dir: str, prefix: str = "train") -> tuple[logging.Logger, str]:
    """
    Create a timed log file + CSV file, attach file + stream handlers.

    Returns
    -------
    logger   : configured Logger instance
    csv_path : path to the metrics CSV file
    """
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{ts}.log")
    csv_path = os.path.join(log_dir, f"{prefix}_metrics_{ts}.csv")

    # Reset handlers so re-importing during tests doesn't duplicate output
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__), csv_path