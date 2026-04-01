"""
utils/inference.py
==================
Sliding-window depth inference shared by both training phases.

Phase 1 (UNet training)    : returns averaged logits (C, D, H, W)
Phase 2 (Survival training): returns argmax mask     (D, H, W)

Single-channel input  (KiTS)   : volume shape (D, H, W)
Multi-channel input   (HECKTOR): volume shape (D, C, H, W)  e.g. (D, 2, H, W) for CT+PT
"""

import torch
from torch.amp import autocast


@torch.no_grad()
def sliding_window_inference(
    model:       torch.nn.Module,
    volume:      torch.Tensor,      # (D, H, W) or (D, C, H, W)
    num_classes: int,
    window:      int          = 16,
    stride:      int          = 8,
    device:      torch.device = torch.device("cpu"),
    use_amp:     bool         = False,
) -> torch.Tensor:
    """
    Run a 3-D model over a full volume with a depth-axis sliding window.
    Overlapping predictions are averaged voxel-wise.

    Accepts both single-channel (D, H, W) and multi-channel (D, C, H, W)
    volumes. The patch sent to the model is always (1, C, win, H, W).

    Returns
    -------
    logits : (num_classes, D, H, W) averaged raw logits on CPU
    """
    # Normalise to (D, C, H, W) regardless of input shape
    if volume.dim() == 3:
        volume = volume.unsqueeze(1)    # (D, H, W) → (D, 1, H, W)

    D, C, H, W = volume.shape
    logits_sum  = torch.zeros(num_classes, D, H, W)
    count       = torch.zeros(D, H, W)

    # Accept either a plain int or a (D, H, W) tuple — only depth matters
    # since the model sees the full H and W every patch.
    if isinstance(window, (tuple, list)):
        window = window[0]
    if isinstance(stride, (tuple, list)):
        stride = stride[0]

    starts = list(range(0, max(1, D - window + 1), stride))
    if not starts or starts[-1] + window < D:
        starts.append(max(0, D - window))

    for z in starts:
        z_end = z + window
        # patch: (D_win, C, H, W) → permute → (C, D_win, H, W) → (1, C, D_win, H, W)
        patch = volume[z:z_end].permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(patch)          # (1, num_classes, D_win, H, W)

        logits_sum[:, z:z_end] += out.squeeze(0).float().cpu()
        count[z:z_end]         += 1.0

    return logits_sum / count.unsqueeze(0).clamp(min=1.0)   # (num_classes, D, H, W)


@torch.no_grad()
def sliding_window_predict(
    model:       torch.nn.Module,
    volume:      torch.Tensor,      # (D, H, W) or (D, C, H, W)
    num_classes: int,
    window:      int          = 16,
    stride:      int          = 8,
    device:      torch.device = torch.device("cpu"),
    in_channels: int          = 1,  # kept for API compat — shape is inferred from volume
) -> torch.Tensor:
    """
    Convenience wrapper: calls sliding_window_inference and returns
    the argmax prediction mask.

    Returns
    -------
    pred_mask : (D, H, W) int64
    """
    logits = sliding_window_inference(
        model       = model,
        volume      = volume,
        num_classes = num_classes,
        window      = window,
        stride      = stride,
        device      = device,
    )
    return logits.argmax(dim=0)     # (D, H, W)