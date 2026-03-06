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

        # Register normalisation buffers so they move with .to(device)
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
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