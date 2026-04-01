"""
models/unet.py
==============
3-D U-Net for volumetric CT segmentation.

Input  : (B, 1, D, H, W)        — single-channel CT, channel dim added by caller
Output : (B, n_classes, D, H, W) — raw logits
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = True):
        super().__init__()
        if trilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = _DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dims (handles odd sizes)
        dZ = x2.size(2) - x1.size(2)
        dY = x2.size(3) - x1.size(3)
        dX = x2.size(4) - x1.size(4)
        x1 = nn.functional.pad(
            x1,
            [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2, dZ // 2, dZ - dZ // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class SimpleUNet3D(nn.Module):
    def __init__(
        self,
        n_classes:     int  = 4,
        base_channels: int  = 32,
        trilinear:     bool = True,
        in_channels:   int  = 1,    # 1 for KiTS (CT only), 2 for HECKTOR (CT+PT)
    ):
        super().__init__()
        b = base_channels
        f = 2 if trilinear else 1

        self.inc   = _DoubleConv(in_channels,  b)
        self.down1 = _Down(b,                  b * 2)
        self.down2 = _Down(b * 2,              b * 4)
        self.down3 = _Down(b * 4,              b * 8)
        self.down4 = _Down(b * 8,              b * 16 // f)

        self.up1   = _Up(b * 16,               b * 8  // f, trilinear)
        self.up2   = _Up(b * 8,                b * 4  // f, trilinear)
        self.up3   = _Up(b * 4,                b * 2  // f, trilinear)
        self.up4   = _Up(b * 2,                b,           trilinear)
        self.outc  = nn.Conv3d(b, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) — C = in_channels (1 for CT-only, 2 for CT+PT)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)