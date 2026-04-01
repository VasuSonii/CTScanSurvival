"""
utils/metrics.py
================
Evaluation metrics shared across training phases.

KiTS21 label hierarchy
-----------------------
  0  background
  1  kidney parenchyma
  2  tumour  (also counts as kidney)
  3  cyst    (also counts as kidney)

Dice is therefore evaluated hierarchically:
  kidney_dice : positive if label ∈ {1, 2, 3}
  tumour_dice : positive if label == 2
  cyst_dice   : positive if label == 3
"""

import torch


def compute_kits_dice(
    logits:  torch.Tensor,   # (B, C, D, H, W)
    targets: torch.Tensor,   # (B, D, H, W) int64
    smooth:  float = 1e-5,
) -> dict:
    """
    Returns
    -------
    dict with keys: kidney_dice, tumour_dice, cyst_dice, mean_dice
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)   # (B, D, H, W)

        def _binary_dice(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> float:
            inter = (pred_pos & gt_pos).float().sum()
            union = pred_pos.float().sum() + gt_pos.float().sum()
            return ((2.0 * inter + smooth) / (union + smooth)).item()

        kidney = _binary_dice(preds >= 1, targets >= 1)
        tumour = _binary_dice(preds == 2, targets == 2)
        cyst   = _binary_dice(preds == 3, targets == 3)

    return {
        "kidney_dice": kidney,
        "tumour_dice": tumour,
        "cyst_dice":   cyst,
        "mean_dice":   (kidney + tumour + cyst) / 3.0,
    }


def concordance_index(
    risk:   torch.Tensor,   # (N,) higher = higher predicted risk
    times:  torch.Tensor,   # (N,) survival times
    events: torch.Tensor,   # (N,) bool / 0-1
) -> float:
    """
    Harrell's C-index.
    Compares all pairs (i, j) where patient i experienced the event
    before patient j.
    """
    n = len(times)
    concordant = discordant = 0

    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j]:
                concordant += (risk[i] > risk[j]).item()
                discordant += (risk[i] < risk[j]).item()

    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def compute_hector_dice(
    pred:   torch.Tensor,   # (D, H, W) int64 — argmax prediction
    target: torch.Tensor,   # (D, H, W) int64 — ground truth
    smooth: float = 1e-5,
) -> dict:
    """
    Patient-level per-class segmentation metrics for HECKTOR.

    Labels
    ------
      0 : background
      1 : tumour  (GTVp — primary tumour)
      2 : lymph node (GTVn — nodal metastasis)

    Metrics are computed at patient level (aggregated over the full D×H×W
    volume) rather than slice level.  Slice-level averaging is misleading
    because slices without tumour have FP=FN=0 by definition, inflating
    recall and precision.

    Returns
    -------
    dict with keys per foreground class (tumour, lymph):
      *_dice      : Dice coefficient
      *_recall    : TP / (TP + FN)  — did we cover the tumour?
      *_precision : TP / (TP + FP)  — how much of prediction is correct?
      *_fp        : raw false positive voxel count
      *_fn        : raw false negative voxel count
    Plus:
      mean_fg_dice : mean Dice over foreground classes
    """
    with torch.no_grad():
        results = {}
        dices   = []

        for label, name in [(1, "tumour"), (2, "lymph")]:
            pred_pos = (pred   == label)
            gt_pos   = (target == label)

            tp = (pred_pos &  gt_pos).float().sum().item()
            fp = (pred_pos & ~gt_pos).float().sum().item()
            fn = (~pred_pos &  gt_pos).float().sum().item()

            dice      = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
            recall    = (tp + smooth) / (tp + fn + smooth)
            precision = (tp + smooth) / (tp + fp + smooth)

            results[f"{name}_dice"]      = dice
            results[f"{name}_recall"]    = recall
            results[f"{name}_precision"] = precision
            results[f"{name}_fp"]        = fp
            results[f"{name}_fn"]        = fn
            dices.append(dice)

        results["mean_fg_dice"] = sum(dices) / len(dices)
    return results