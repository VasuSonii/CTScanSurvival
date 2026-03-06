"""
utils/inference.py
==================
Sliding-window depth inference shared by both training phases.

Phase 1 (UNet training)   : returns averaged logits (C, D, H, W)
Phase 2 (Survival training): returns argmax mask    (D, H, W)
"""

import torch
from torch.amp import autocast


@torch.no_grad()
def sliding_window_inference(
    model:       torch.nn.Module,
    volume:      torch.Tensor,          # (D, H, W) float32 [0, 1]
    num_classes: int,
    window:      int          = 16,
    stride:      int          = 8,
    device:      torch.device = torch.device("cpu"),
    use_amp:     bool         = False,
) -> torch.Tensor:
    """
    Run a 3-D model over a full volume with a depth-axis sliding window.
    Overlapping predictions are averaged voxel-wise.

    Returns
    -------
    logits : (C, D, H, W) averaged raw logits on CPU
    """
    D, H, W    = volume.shape
    logits_sum = torch.zeros(num_classes, D, H, W)
    count      = torch.zeros(D, H, W)

    starts = list(range(0, max(1, D - window + 1), stride))
    if not starts or starts[-1] + window < D:
        starts.append(max(0, D - window))

    for z in starts:
        z_end = z + window
        patch = volume[z:z_end].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,win,H,W)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(patch)                                          # (1,C,win,H,W)

        logits_sum[:, z:z_end] += out.squeeze(0).float().cpu()
        count[z:z_end]         += 1.0

    return logits_sum / count.unsqueeze(0).clamp(min=1.0)  # (C, D, H, W)


@torch.no_grad()
def sliding_window_predict(
    model:       torch.nn.Module,
    volume:      torch.Tensor,
    num_classes: int,
    window:      int          = 16,
    stride:      int          = 8,
    device:      torch.device = torch.device("cpu"),
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
    return logits.argmax(dim=0)   # (D, H, W)