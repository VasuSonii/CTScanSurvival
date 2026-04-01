"""
models/clinical_mlp.py
=======================
Lightweight MLP that maps a clinical feature vector to a fixed-size embedding
for fusion with the imaging-derived patient embedding.

Architecture
------------
    clinical_dim → Linear → BN → GELU → Dropout
                 → Linear → BN → GELU → Dropout
                 → Linear → output_dim

LayerNorm is used (not BatchNorm) because we process one patient at a
time — BatchNorm requires batch size > 1 during training and would crash
with the per-patient loop.
"""

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """
    Parameters
    ----------
    input_dim   : number of clinical features after preprocessing
    output_dim  : embedding dimension to concatenate with imaging features
    hidden_dims : sizes of intermediate layers (default: [256, 128])
    dropout     : dropout probability applied after each hidden activation
    """

    def __init__(
        self,
        input_dim:   int,
        output_dim:  int         = 128,
        hidden_dims: list[int]   = None,
        dropout:     float       = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim) or (input_dim,) clinical feature tensor

        Returns
        -------
        (B, output_dim) or (output_dim,) embedding
        """
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        out = self.net(x)
        return out.squeeze(0) if unbatched else out