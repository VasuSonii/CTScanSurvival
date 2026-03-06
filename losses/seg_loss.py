"""
losses/seg_loss.py
==================
Combined Cross-Entropy + Soft-Dice loss for multi-class segmentation.

KiTS21 label hierarchy (handled by the caller via compute_kits_dice):
  0 background  |  1 kidney  |  2 tumour  |  3 cyst
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEDiceLoss(nn.Module):
    """
    Parameters
    ----------
    ce_weight         : weight of CE term; Dice weight = 1 - ce_weight
    smooth            : Laplace smoothing for Dice
    class_weights     : optional per-class CE weights (list of floats)
    ignore_background : exclude class-0 from Dice when it has no GT voxels
    """

    def __init__(
        self,
        ce_weight:         float       = 0.5,
        smooth:            float       = 1e-5,
        class_weights:     list | None = None,
        ignore_background: bool        = True,
    ):
        super().__init__()
        if not 0.0 <= ce_weight <= 1.0:
            raise ValueError(f"ce_weight must be in [0, 1], got {ce_weight}")

        self.ce_weight         = ce_weight
        self.smooth            = smooth
        self.ignore_background = ignore_background

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float),
            )
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C, D, H, W)
        targets: torch.Tensor,   # (B, D, H, W) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss, ce_loss, dice_loss, debug_stats
        """
        targets     = targets.long()
        num_classes = logits.shape[1]

        # ── Cross-entropy ─────────────────────────────────────────────────
        ce_loss = self.ce(logits, targets)

        # ── Soft Dice ─────────────────────────────────────────────────────
        probs  = F.softmax(logits, dim=1).clamp(1e-7, 1 - 1e-7)
        onehot = F.one_hot(targets, num_classes).movedim(-1, 1).float()  # (B,C,D,H,W)

        dims         = (0, 2, 3, 4)
        intersection = torch.sum(probs * onehot, dim=dims)           # (C,)
        cardinality  = torch.sum(probs + onehot, dim=dims)           # (C,)
        dice_scores  = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Exclude classes absent in this batch
        gt_sum     = torch.sum(onehot, dim=dims)
        valid_mask = gt_sum > 0
        if self.ignore_background and num_classes > 1:
            valid_mask[0] = False

        valid_dice = dice_scores[valid_mask]
        dice_loss  = (
            torch.tensor(0.0, device=logits.device)
            if valid_dice.numel() == 0
            else 1.0 - valid_dice.mean()
        )

        # ── Debug stats ───────────────────────────────────────────────────
        debug_stats = {
            "max_bg_prob":    probs[:, 0].max().item(),
            "max_fg_prob":    probs[:, 1:].max().item() if num_classes > 1 else 0.0,
            "intersect":      intersection.detach().cpu().tolist(),
            "cardinality":    cardinality.detach().cpu().tolist(),
            "dice_per_class": dice_scores.detach().cpu().tolist(),
        }

        total_loss = self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss
        return total_loss, ce_loss, dice_loss, debug_stats