"""
models/comorbidity_classifier.py
=================================
Small MLP classifier that maps an OmniRad patient embedding to 5 binary
comorbidity predictions.

Architecture
------------
    embed_dim (768) → Linear → LayerNorm → GELU → Dropout
                    → Linear → LayerNorm → GELU → Dropout
                    → Linear(5)   ← raw logits, no sigmoid

Output is raw logits — use BCEWithLogitsLoss during training,
torch.sigmoid() to get probabilities at inference.

Why small?
----------
The task is predicting 5 rare binary labels from a 768-dim embedding
trained for a different purpose (survival).  A large classifier would
memorise the training set.  Two hidden layers of 128 units is sufficient
to probe what the embedding encodes.
"""

import torch
import torch.nn as nn

from data.comorbidity_labels import N_LABELS


class ComorbidityClassifier(nn.Module):
    """
    Parameters
    ----------
    embed_dim   : OmniRad embedding dimension (default 768)
    hidden_dim  : hidden layer width
    dropout     : dropout probability
    n_labels    : number of binary outputs (default = N_LABELS = 5)
    """

    def __init__(
        self,
        embed_dim:  int   = 768,
        hidden_dim: int   = 128,
        dropout:    float = 0.3,
        n_labels:   int   = N_LABELS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),   # raw logits
        )
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
        x : (B, embed_dim) or (embed_dim,) patient embedding

        Returns
        -------
        logits : (B, n_labels) or (n_labels,) — raw logits
        """
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        out = self.net(x)
        return out.squeeze(0) if unbatched else out