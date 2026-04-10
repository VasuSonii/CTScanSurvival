"""
data/comorbidity_labels.py
===========================
Loads 5 binary comorbidity labels from kits.json for each patient.

Target labels
-------------
  0  chronic_kidney_disease
  1  mild_liver_disease
  2  moderate_to_severe_liver_disease
  3  diabetes_mellitus_with_end_organ_damage
  4  peripheral_vascular_disease

These are all rare conditions with significant class imbalance.
Prevalence is computed and logged so BCE class weights can be set
appropriately.

Usage
-----
    from data.comorbidity_labels import load_comorbidity_labels

    labels, weights, prevalence = load_comorbidity_labels(
        metadata_path = "/path/to/kits.json",
        case_ids      = train_ids + val_ids,
    )
    # labels       : {case_id: Tensor(5,) float32  0 or 1}
    # weights      : Tensor(5,) — pos_weight for BCEWithLogitsLoss
    # prevalence   : {label_name: float}  fraction of positive cases
"""

import json
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Canonical label order — do not change without updating classifier output dim
COMORBIDITY_LABELS = [
    "chronic_kidney_disease",
    "mild_liver_disease",
    "moderate_to_severe_liver_disease",
    "diabetes_mellitus_with_end_organ_damage",
    "peripheral_vascular_disease",
]
N_LABELS = len(COMORBIDITY_LABELS)


def load_comorbidity_labels(
    metadata_path: str,
    case_ids:      list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, float]]:
    """
    Load binary comorbidity labels from kits.json.

    Parameters
    ----------
    metadata_path : path to kits.json
    case_ids      : list of case IDs to load (train + val combined)

    Returns
    -------
    labels     : {case_id: Tensor(N_LABELS,) float32}  — 0.0 or 1.0
    pos_weight : Tensor(N_LABELS,)  — (n_neg / n_pos) per label for
                 BCEWithLogitsLoss(pos_weight=...) to handle imbalance
    prevalence : {label_name: float}  — fraction of positive cases
    """
    with open(metadata_path) as f:
        data = json.load(f)
    meta = {entry["case_id"]: entry for entry in data}

    labels:     dict[str, torch.Tensor] = {}
    counts_pos = torch.zeros(N_LABELS)
    counts_tot = 0

    missing = []
    for cid in case_ids:
        if cid not in meta:
            missing.append(cid)
            continue

        comorbidities = meta[cid].get("comorbidities", {})
        vec = torch.tensor(
            [float(bool(comorbidities.get(label, False)))
             for label in COMORBIDITY_LABELS],
            dtype=torch.float32,
        )
        labels[cid] = vec
        counts_pos += vec
        counts_tot += 1

    if missing:
        logger.warning(f"Missing comorbidity data for {len(missing)} cases: {missing}")

    # Prevalence
    prevalence = {
        label: (counts_pos[i].item() / max(counts_tot, 1))
        for i, label in enumerate(COMORBIDITY_LABELS)
    }

    # BCEWithLogitsLoss pos_weight = n_neg / n_pos (clamped to avoid inf)
    counts_neg = counts_tot - counts_pos
    pos_weight = (counts_neg / counts_pos.clamp(min=1.0)).clamp(max=50.0)

    for i, label in enumerate(COMORBIDITY_LABELS):
        n_pos = int(counts_pos[i].item())
        logger.info(
            f"  {label:<45s}  "
            f"pos={n_pos:3d}/{counts_tot}  "
            f"prevalence={prevalence[label]:.1%}  "
            f"pos_weight={pos_weight[i]:.1f}"
        )

    return labels, pos_weight, prevalence