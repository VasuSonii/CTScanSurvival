"""
utils/kfold.py
==============
Stratified k-fold split generation for survival experiments.

Stratification is by event status (dead / censored) so each fold
maintains the same event rate as the full dataset.  With small datasets
(<300 patients) this matters: a random split could put all events in one
fold by chance.

Usage
-----
    from utils.kfold import make_kfold_splits, load_events_from_metadata

    events = load_events_from_metadata(metadata_path, case_ids)
    folds  = make_kfold_splits(case_ids, events, n_folds=5, seed=42)

    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        print(f"Fold {fold_idx}: train={len(train_ids)}, val={len(val_ids)}")
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def load_events_from_metadata(
    metadata_path: str,
    case_ids:      list[str],
) -> np.ndarray:
    """
    Load binary event labels (1=dead, 0=censored) for the given case IDs.

    Supports both:
      - KiTS23 JSON format  (list of dicts with 'case_id' and 'vital_status')
      - HECKTOR CSV format  (CSV with 'PatientID' and 'Relapse' columns)

    Parameters
    ----------
    metadata_path : path to kits.json or metadata.csv
    case_ids      : ordered list of case IDs — events returned in same order

    Returns
    -------
    events : (N,) int array, 1=event occurred, 0=censored
    """
    if metadata_path.endswith(".csv"):
        df = pd.read_csv(metadata_path)
        # HECKTOR format
        if "PatientID" in df.columns:
            df = df.set_index("PatientID")
            events = np.array([int(df.loc[cid, "Relapse"]) for cid in case_ids])
        else:
            raise ValueError(f"Unknown CSV format in {metadata_path}")
    else:
        # KiTS JSON format
        with open(metadata_path) as f:
            data = json.load(f)
        meta = {entry["case_id"]: entry for entry in data}
        events = np.array([
            1 if meta[cid]["vital_status"] == "dead" else 0
            for cid in case_ids
        ])

    event_rate = events.mean()
    logger.info(
        f"Event rate: {event_rate:.1%}  "
        f"({events.sum()} events / {len(events)} patients)"
    )
    return events


def make_kfold_splits(
    case_ids: list[str],
    events:   np.ndarray,
    n_folds:  int = 5,
    seed:     int = 42,
) -> list[tuple[list[str], list[str]]]:
    """
    Generate stratified k-fold splits.

    Parameters
    ----------
    case_ids : all case IDs in the dataset (train + val combined)
    events   : binary event labels, same order as case_ids
    n_folds  : number of folds
    seed     : random seed for reproducibility

    Returns
    -------
    list of (train_ids, val_ids) tuples, one per fold
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if len(case_ids) < n_folds:
        raise ValueError(
            f"Fewer patients ({len(case_ids)}) than folds ({n_folds})"
        )

    ids_arr = np.array(case_ids)
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids_arr, events)):
        train_ids = ids_arr[train_idx].tolist()
        val_ids   = ids_arr[val_idx].tolist()

        fold_events = events[val_idx]
        logger.info(
            f"Fold {fold_idx}: train={len(train_ids)}  val={len(val_ids)}  "
            f"val_event_rate={fold_events.mean():.1%}"
        )
        folds.append((train_ids, val_ids))

    return folds


def get_all_case_ids(split_file: str) -> list[str]:
    """
    Read ALL case IDs from a split file (both train and val splits combined).
    Used to build the full pool before generating k-fold splits.

    Supports the standard KiTS/HECKTOR split format:
      {"train": [...], "val": [...]}
    and nested Task format:
      {"task1": {"train": [...], "val": [...]}, ...}
    """
    with open(split_file) as f:
        splits = json.load(f)

    # Flatten all lists of IDs found anywhere in the split file
    all_ids: list[str] = []
    def _collect(obj):
        if isinstance(obj, list):
            all_ids.extend(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
    _collect(splits)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for cid in all_ids:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)

    logger.info(f"Total unique case IDs from split file: {len(unique)}")
    return unique