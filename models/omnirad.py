"""
models/omnirad.py
=================
OmniRad ViT encoder wrapper.

3-channel slice convention
--------------------------
  ch0 : CT slice          (float32, [0, 1])
  ch1 : predicted mask    (float32, values 0-3 normalised → [0, 1])
  ch2 : CT slice (repeat) (float32, [0, 1])

All channels are resized to 224×224 and normalised with ImageNet
mean/std so the pretrained weights transfer correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_IMG_SIZE      = 224


class OmniRadEncoder(nn.Module):
    """
    Frozen OmniRad ViT encoder.

    Public API
    ----------
    encode_pil(image)              — PIL.Image → (1, embed_dim)
    encode_tensor_slice(t3)        — (3, H, W) float tensor → (1, embed_dim)
    encode_volume(ct, mask, ...)   — (D, H, W) CT + mask → (D, embed_dim) on CPU
    build_slice_tensor(ct, mask)   — static: assemble the 3-channel input for one slice
    """

    def __init__(
        self,
        device: torch.device | None = None,
        frozen: bool = True,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            "hf_hub:Snarcy/OmniRad-base",
            pretrained=True,
            num_classes=0,          # raw CLS embeddings
        )
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.model.eval().to(self.device)

        # PIL pipeline (unchanged from original)
        self.pil_transform = transforms.Compose([
            transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Create buffers directly on self.device.
        # register_buffer() alone is not enough here because OmniRadEncoder
        # itself is never moved with .to(device) — only self.model is — so
        # buffers created on CPU would stay on CPU while slices are on GPU.
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=self.device).view(3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _preprocess(self, t3: torch.Tensor) -> torch.Tensor:
        """(3, H, W) float [0,1] → (1, 3, 224, 224) normalised."""
        t3 = t3.to(self.device)
        t3 = resize(t3, [_IMG_SIZE, _IMG_SIZE],
                    interpolation=InterpolationMode.BILINEAR, antialias=True)
        t3 = (t3 - self._mean) / self._std
        return t3.unsqueeze(0)

    @staticmethod
    def build_slice_tensor(
        ct_slice:    torch.Tensor,   # (H, W) float32 [0, 1]
        mask_slice:  torch.Tensor,   # (H, W) int64 or float32, values 0-3
        num_classes: int = 4,
    ) -> torch.Tensor:
        """Return a (3, H, W) tensor ready for the encoder."""
        mask_norm = mask_slice.float() / max(num_classes - 1, 1)
        return torch.stack(
            [ct_slice.float(), mask_norm, ct_slice.float()], dim=0
        )

    # ── Public API ────────────────────────────────────────────────────────

    def encode_pil(self, image: Image.Image) -> torch.Tensor:
        """PIL → (1, embed_dim)."""
        x = self.pil_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(x)

    def encode_tensor_slice(self, t3: torch.Tensor) -> torch.Tensor:
        """(3, H, W) float [0,1] → (1, embed_dim)."""
        with torch.no_grad():
            return self.model(self._preprocess(t3))

    @torch.no_grad()
    def encode_volume(
        self,
        ct:          torch.Tensor,   # (D, H, W) float32 [0, 1]
        mask:        torch.Tensor,   # (D, H, W) int64 / float32, values 0-3
        num_classes: int = 4,
        batch_size:  int = 16,       # slices per forward pass — tune to VRAM
    ) -> torch.Tensor:
        """
        Encode every slice of a CT volume.

        Returns
        -------
        embeddings : (D, embed_dim) on CPU
        """
        D         = ct.shape[0]
        all_embs  = []

        for start in range(0, D, batch_size):
            end    = min(start + batch_size, D)
            slices = torch.stack(
                [self.build_slice_tensor(ct[i], mask[i], num_classes)
                 for i in range(start, end)],
                dim=0,
            ).to(self.device)                           # (B, 3, H, W)

            slices = F.interpolate(slices, size=(_IMG_SIZE, _IMG_SIZE),
                                   mode="bilinear", align_corners=False)
            slices = (slices - self._mean.unsqueeze(0)) / self._std.unsqueeze(0)

            all_embs.append(self.model(slices).cpu())  # (B, embed_dim)

        return torch.cat(all_embs, dim=0)              # (D, embed_dim)

    @torch.no_grad()
    def encode_volume_rgb(
        self,
        ct_rgb:     torch.Tensor,   # (D, 3, H, W) float32 [0, 1] — pre-built RGB
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Encode a pre-built 3-channel HU-windowed CT volume.

        Unlike encode_volume(), this method receives the 3 channels already
        assembled (e.g. by KitsRGBDataset) so no CT+mask interleaving is
        needed.  It only applies spatial resize and ImageNet normalisation
        before passing slices through the ViT.

        Parameters
        ----------
        ct_rgb     : (D, 3, H, W) float32, each channel already in [0, 1]
        batch_size : slices per forward pass — tune to VRAM

        Returns
        -------
        embeddings : (D, embed_dim) on CPU
        """
        D        = ct_rgb.shape[0]
        all_embs = []

        for start in range(0, D, batch_size):
            end    = min(start + batch_size, D)
            slices = ct_rgb[start:end].to(self.device)           # (B, 3, H, W)

            slices = F.interpolate(
                slices, size=(_IMG_SIZE, _IMG_SIZE),
                mode="bilinear", align_corners=False,
            )
            slices = (slices - self._mean.unsqueeze(0)) / self._std.unsqueeze(0)

            all_embs.append(self.model(slices).cpu())            # (B, embed_dim)

        return torch.cat(all_embs, dim=0)                        # (D, embed_dim)

    @torch.no_grad()
    def encode_volume_ct_pt(
        self,
        ct:          torch.Tensor,   # (D, H, W) float32 [0, 1]
        pt:          torch.Tensor,   # (D, H, W) float32 [0, 1]
        mask:        torch.Tensor,   # (D, H, W) int64 / float32, values 0-1
        num_classes: int = 2,        # HECKTOR: background + tumour
        batch_size:  int = 16,
    ) -> torch.Tensor:
        """
        Encode a CT+PT volume using OmniRad.

        3-channel layout for HECKTOR: [CT, PT, mask_norm]
        All three channels carry real signal — CT for anatomy, PT for
        metabolic activity, mask for tumour localisation.

        Compare with KiTS encode_volume which uses [CT, mask_norm, CT]
        because only CT is available (PT channel duplicates CT).

        Parameters
        ----------
        ct         : (D, H, W) CT  normalised to [0, 1]
        pt         : (D, H, W) PT  normalised to [0, 1]
        mask       : (D, H, W) segmentation mask — predicted or ground-truth
        num_classes: number of segmentation classes (for mask normalisation)
        batch_size : slices per forward pass

        Returns
        -------
        embeddings : (D, embed_dim) on CPU
        """
        D        = ct.shape[0]
        all_embs = []
        mask_norm_scale = max(num_classes - 1, 1)

        for start in range(0, D, batch_size):
            end   = min(start + batch_size, D)

            # Build (B, 3, H, W): ch0=CT, ch1=PT, ch2=mask_norm
            batch = torch.stack([
                torch.stack([
                    ct[i].float(),
                    pt[i].float(),
                    mask[i].float() / mask_norm_scale,
                ], dim=0)
                for i in range(start, end)
            ], dim=0).to(self.device)                            # (B, 3, H, W)

            batch = F.interpolate(
                batch, size=(_IMG_SIZE, _IMG_SIZE),
                mode="bilinear", align_corners=False,
            )
            batch = (batch - self._mean.unsqueeze(0)) / self._std.unsqueeze(0)

            all_embs.append(self.model(batch).cpu())             # (B, embed_dim)

        return torch.cat(all_embs, dim=0)                        # (D, embed_dim)

# ─── Slice pooling ────────────────────────────────────────────────────────────

class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling over a variable-length set of slice embeddings.

    Based on: Ilse et al., "Attention-based Deep Multiple Instance Learning"
    (ICML 2018).  https://arxiv.org/abs/1802.04712

    Given D slice embeddings H = {h_1, ..., h_D} ∈ R^(D × L):

        a_i = softmax_i( w^T (tanh(V h_i) ⊙ sigmoid(U h_i)) )
        z   = Σ_i a_i h_i                     ∈ R^L

    The gating mechanism (sigmoid branch) lets the network suppress
    uninformative slices rather than just re-weighting them.

    This module is **trainable** — it should be instantiated and owned by the
    training script, included in the optimizer, and saved in the checkpoint
    separately from the frozen OmniRad encoder.

    Parameters
    ----------
    embed_dim   : dimensionality of each slice embedding (L)
    hidden_size : inner dimension of the attention projection (default 128)
    dropout     : dropout applied to both attention branches (default 0.25)

    Inputs
    ------
    h : (D, L) or (B, D, L) — slice embeddings for one patient (or a batch)

    Outputs
    -------
    z       : (L,) or (B, L) — pooled patient embedding
    weights : (D,) or (B, D) — normalised attention weights (for inspection)
    """

    def __init__(
        self,
        embed_dim:   int,
        hidden_size: int   = 128,
        dropout:     float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size

        # Tanh branch  V ∈ R^(H × L)
        self.tanh_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=True),
            nn.Tanh(),
            nn.Dropout(p=dropout),
        )
        # Sigmoid gate  U ∈ R^(H × L)
        self.gate_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=True),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
        )
        # Attention projection  w ∈ R^(1 × H)
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (D, L) for a single patient, or (B, D, L) for a batch

        Returns
        -------
        z       : (L,) or (B, L)
        weights : (D,) or (B, D)  — softmax-normalised attention scores
        """
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)   # → (1, D, L)

        # Gated attention score: (B, D, H) → (B, D, 1)
        score = self.attention(self.tanh_branch(h) * self.gate_branch(h))
        weights = torch.softmax(score, dim=1)    # (B, D, 1)

        z = (weights * h).sum(dim=1)             # (B, L)

        if unbatched:
            return z.squeeze(0), weights.squeeze(0).squeeze(-1)  # (L,), (D,)
        return z, weights.squeeze(-1)            # (B, L), (B, D)