"""
data/dataset.py
===============
KiTS21 dataset.  Returns:
  - train mode : 16-slice tumour-centred crop  (D=16, H, W)
  - val   mode : full volume                   (D,    H, W)

Each sample always includes event flag and survival time for downstream
survival training.
"""

import json
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


class KitsDataset(Dataset):
    def __init__(
        self,
        rootdir:        str,
        target_spacing: tuple,          # (X, Y, Z) SimpleITK convention
        target_shape:   tuple,          # (H, W) after spatial resize
        split_file:     str,
        metadata_path:  str,
        p:              float = 0.8,    # prob of tumour-centred depth crop
        mode:           str   = "train",
        crop_depth:     int   = 16,
    ):
        self.rootdir        = rootdir
        self.target_spacing = target_spacing
        self.target_shape   = target_shape
        self.split_file     = split_file
        self.metadata_path  = metadata_path
        self.p              = p
        self.mode           = mode
        self.crop_depth     = crop_depth

        self.metadata: dict = {}
        self._load_cases()

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_cases(self) -> None:
        with open(self.split_file, "r") as f:
            splits = json.load(f)
        self.cases = splits.get(self.mode, splits.get("train"))

        with open(self.metadata_path, "r") as f:
            for entry in json.load(f):
                self.metadata[entry["case_id"]] = entry

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        caseid = self.cases[index]

        image = sitk.ReadImage(os.path.join(self.rootdir, caseid, "imaging.nii.gz"))
        mask  = sitk.ReadImage(os.path.join(self.rootdir, caseid, "aggregated_MAJ_seg.nii.gz"))

        image = sitk.DICOMOrient(image, "RAS")
        mask  = sitk.DICOMOrient(mask,  "RAS")

        image, mask = self._resample(image, mask)
        image, mask = self._to_tensors(image, mask)
        image, mask = self._spatial_resize(image, mask)  # H, W → target_shape

        if self.mode == "train":
            image, mask = self._train_crop(image, mask)  # (crop_depth, H, W)
        # val: full volume (D, H, W) — sliding window handled in training loop

        event, survival_time = self._get_survival(caseid)

        return {
            "ct":            image,
            "mask":          mask,
            "caseid":        caseid,
            "event":         torch.tensor(event,         dtype=torch.bool),
            "survival_time": torch.tensor(survival_time, dtype=torch.float32),
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _resample(
        self,
        image: sitk.Image,
        mask:  sitk.Image,
    ) -> tuple[sitk.Image, sitk.Image]:
        original_size    = image.GetSize()
        original_spacing = image.GetSpacing()

        new_size = [
            int(round(osz * osp / tsp))
            for osz, osp, tsp in zip(original_size, original_spacing, self.target_spacing)
        ]

        def _do_resample(itk_img: sitk.Image, is_mask: bool) -> sitk.Image:
            r = sitk.ResampleImageFilter()
            r.SetSize(new_size)
            r.SetOutputSpacing(self.target_spacing)
            r.SetOutputOrigin(itk_img.GetOrigin())
            r.SetOutputDirection(itk_img.GetDirection())
            if is_mask:
                r.SetInterpolator(sitk.sitkNearestNeighbor)
                r.SetDefaultPixelValue(0)
            else:
                r.SetInterpolator(sitk.sitkLinear)
                r.SetDefaultPixelValue(-1000)
            return r.Execute(itk_img)

        return _do_resample(image, False), _do_resample(mask, True)

    def _to_tensors(
        self,
        image: sitk.Image,
        mask:  sitk.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_arr  = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)
        mask_arr = sitk.GetArrayFromImage(mask).astype(np.int64)      # (D, H, W)

        # HU windowing → [0, 1]
        img_arr = np.clip(img_arr, -200.0, 300.0)
        img_arr = (img_arr + 200.0) / 500.0

        return torch.from_numpy(img_arr), torch.from_numpy(mask_arr)

    def _spatial_resize(
        self,
        image: torch.Tensor,   # (D, H, W)
        mask:  torch.Tensor,   # (D, H, W) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(
            image.unsqueeze(0),        # (1, D, H, W)
            list(self.target_shape),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)

        mask = TF.resize(
            mask.float().unsqueeze(0),
            list(self.target_shape),
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0).long()

        return image, mask

    def _train_crop(
        self,
        image: torch.Tensor,
        mask:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth = image.shape[0]
        crop  = self.crop_depth

        nonzero_slices = (mask != 0).any(dim=(1, 2))
        indices        = torch.where(nonzero_slices)[0]

        if len(indices) > 0 and torch.rand(()) < self.p:
            center = indices[random.randint(0, len(indices) - 1)].item()
            z_min  = max(0, center - crop + 1)
            z_max  = min(center, depth - crop)
            z = (
                max(0, min(center, depth - crop))
                if z_max < z_min
                else random.randint(z_min, z_max)
            )
        else:
            z = random.randint(0, max(0, depth - crop))

        return image[z:z + crop], mask[z:z + crop]

    def _get_survival(self, caseid: str) -> tuple[bool, float]:
        """
        Returns
        -------
        event         : True if patient died (vital_status == 'dead')
        survival_time : days after surgery (vital_days_after_surgery)
        """
        meta          = self.metadata[caseid]
        event         = meta["vital_status"] == "dead"
        survival_time = meta.get("vital_days_after_surgery") or 0.0
        return bool(event), float(survival_time)