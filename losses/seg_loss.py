"""
losses/seg_loss.py
==================
Two segmentation losses for the UNet training phase.

CEDiceLoss    — CE + Soft Dice.  Penalises FP and FN equally.
                Good for KiTS where exact boundary quality matters
                (Dice metric is evaluated hierarchically).

CETverskyLoss — CE + Tversky.  Asymmetric: penalises FN >> FP.
                Good for HECKTOR where the mask is used as an OmniRad
                input channel rather than evaluated directly.  The model
                learns to over-predict slightly, covering the full tumour
                region rather than tight boundaries.  This gives OmniRad
                a reliable spatial context signal.

Both return (total_loss, primary_loss, seg_loss, debug_stats) so the
calling code is identical regardless of which loss is used.

Select via config:
    loss_type = "dice"    → CEDiceLoss
    loss_type = "tversky" → CETverskyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_weight_buffer(class_weights):
    """Accept list/tuple or pre-built tensor without triggering UserWarning."""
    if class_weights is None:
        return None
    if isinstance(class_weights, torch.Tensor):
        return class_weights.float().clone().detach()
    return torch.tensor(class_weights, dtype=torch.float)


# ── CE + Soft Dice ─────────────────────────────────────────────────────────────

class CEDiceLoss(nn.Module):
    """
    CE + Soft Dice.  FP and FN penalised equally.

    Use for:  KiTS (exact segmentation quality matters)

    Parameters
    ----------
    ce_weight         : weight of CE; Dice weight = 1 - ce_weight
    smooth            : Laplace smoothing for Dice
    class_weights     : optional per-class CE weights (list or tensor)
    ignore_background : exclude class-0 from Dice mean
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

        w = _build_weight_buffer(class_weights)
        if w is not None:
            self.register_buffer("class_weights", w)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C, D, H, W)
        targets: torch.Tensor,   # (B, D, H, W) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Returns (total_loss, ce_loss, dice_loss, debug_stats)."""
        targets     = targets.long()
        num_classes = logits.shape[1]

        ce_loss = self.ce(logits, targets)

        probs  = F.softmax(logits, dim=1).clamp(1e-7, 1 - 1e-7)
        onehot = F.one_hot(targets, num_classes).movedim(-1, 1).float()

        dims         = (0, 2, 3, 4)
        intersection = torch.sum(probs * onehot, dim=dims)
        cardinality  = torch.sum(probs + onehot, dim=dims)
        dice_scores  = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

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

        debug_stats = {
            "max_bg_prob":    probs[:, 0].max().item(),
            "max_fg_prob":    probs[:, 1:].max().item() if num_classes > 1 else 0.0,
            "dice_per_class": dice_scores.detach().cpu().tolist(),
        }

        total_loss = self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss
        return total_loss, ce_loss, dice_loss, debug_stats


# ── CE + Tversky ───────────────────────────────────────────────────────────────

class CETverskyLoss(nn.Module):
    """
    CE + Tversky loss.  Asymmetric: penalises FN >> FP.

    Use for:  HECKTOR (mask used as OmniRad input, not evaluated directly).
              The model learns to predict a broader region around the tumour,
              ensuring OmniRad always sees the full tumour context.

    Tversky index:
        T = (TP + smooth) / (TP + α·FP + β·FN + smooth)

    With α < β the model is less punished for over-predicting (FP) than for
    missing tumour voxels (FN), resulting in higher recall at the cost of
    slightly lower precision.

    Parameters
    ----------
    ce_weight         : weight of CE; Tversky weight = 1 - ce_weight
    alpha             : FP weight  (default 0.2 — low penalty for over-predict)
    beta              : FN weight  (default 0.8 — high penalty for missing tumour)
    smooth            : smoothing to avoid division by zero
    class_weights     : optional per-class CE weights (list or tensor)
    ignore_background : exclude class-0 from Tversky mean
    """

    def __init__(
        self,
        ce_weight:         float       = 0.5,
        alpha:             float       = 0.2,
        beta:              float       = 0.8,
        smooth:            float       = 1e-5,
        class_weights:     list | None = None,
        ignore_background: bool        = True,
    ):
        super().__init__()
        if not 0.0 <= ce_weight <= 1.0:
            raise ValueError(f"ce_weight must be in [0, 1], got {ce_weight}")
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta should equal 1.0, got {alpha + beta:.4f}")
        self.ce_weight         = ce_weight
        self.alpha             = alpha
        self.beta              = beta
        self.smooth            = smooth
        self.ignore_background = ignore_background

        w = _build_weight_buffer(class_weights)
        if w is not None:
            self.register_buffer("class_weights", w)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C, D, H, W)
        targets: torch.Tensor,   # (B, D, H, W) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Returns (total_loss, ce_loss, tversky_loss, debug_stats)."""
        targets     = targets.long()
        num_classes = logits.shape[1]

        ce_loss = self.ce(logits, targets)

        probs  = F.softmax(logits, dim=1).clamp(1e-7, 1 - 1e-7)
        onehot = F.one_hot(targets, num_classes).movedim(-1, 1).float()

        dims      = (0, 2, 3, 4)
        true_pos  = torch.sum(probs * onehot,           dim=dims)
        false_pos = torch.sum(probs * (1.0 - onehot),   dim=dims)
        false_neg = torch.sum((1.0 - probs) * onehot,   dim=dims)

        tversky_scores = (true_pos + self.smooth) / (
            true_pos
            + self.alpha * false_pos
            + self.beta  * false_neg
            + self.smooth
        )

        gt_sum     = torch.sum(onehot, dim=dims)
        valid_mask = gt_sum > 0
        if self.ignore_background and num_classes > 1:
            valid_mask[0] = False

        valid_tversky = tversky_scores[valid_mask]
        tversky_loss  = (
            torch.tensor(0.0, device=logits.device)
            if valid_tversky.numel() == 0
            else 1.0 - valid_tversky.mean()
        )

        debug_stats = {
            "max_bg_prob":     probs[:, 0].max().item(),
            "max_fg_prob":     probs[:, 1:].max().item() if num_classes > 1 else 0.0,
            "true_pos":        true_pos.detach().cpu().tolist(),
            "false_pos":       false_pos.detach().cpu().tolist(),
            "false_neg":       false_neg.detach().cpu().tolist(),
            "tversky_per_cls": tversky_scores.detach().cpu().tolist(),
        }

        total_loss = self.ce_weight * ce_loss + (1.0 - self.ce_weight) * tversky_loss
        return total_loss, ce_loss, tversky_loss, debug_stats


# ── Factory ───────────────────────────────────────────────────────────────────

def build_seg_loss(cfg) -> nn.Module:
    """
    Instantiate the right loss from a UNet config.

    Always call .to(device) on the returned module so the class_weights
    buffer lands on the correct device before the first forward pass.

    cfg must have:
      loss_type     : "dice" | "tversky"
      ce_weight     : float
      class_weights : list | None

    CETverskyLoss also reads:
      tversky_alpha, tversky_beta
    """
    # Pass the raw list — _build_weight_buffer inside each class handles
    # list vs tensor safely.  Device placement is done by the caller via
    # .to(device) on the returned module.
    w = list(cfg.class_weights) if cfg.class_weights else None

    if cfg.loss_type == "tversky":
        return CETverskyLoss(
            ce_weight     = cfg.ce_weight,
            alpha         = cfg.tversky_alpha,
            beta          = cfg.tversky_beta,
            class_weights = w,
        )
    elif cfg.loss_type == "dice":
        return CEDiceLoss(
            ce_weight     = cfg.ce_weight,
            class_weights = w,
        )
    else:
        raise ValueError(
            f"Unknown loss_type='{cfg.loss_type}'. Choose 'dice' or 'tversky'."
        )