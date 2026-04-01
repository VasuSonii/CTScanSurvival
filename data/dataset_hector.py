"""
data/dataset_hector.py
=======================
HECKTOR 2025 dataset classes.

Task 1  — CT + PT + tumour mask  → used to train the 2-channel UNet
Task 2  — CT + PT only           → used for survival training

File layout per patient (both tasks)
--------------------------------------
  <task_dir>/<PatientID>/<PatientID>__CT.nii.gz
  <task_dir>/<PatientID>/<PatientID>__PT.nii.gz
  <task_dir>/<PatientID>/<PatientID>.nii.gz      ← mask (Task 1 only)

Split file structure
--------------------
  {
    "task1": {"train": [...], "val": [...]},
    "task2": {"train": [...], "val": [...]}
  }

Metadata (CSV)
--------------
  PatientID, CenterID, Age, Gender, ..., Relapse, RFS
  RFS is in days.  Relapse is 0/1.

Normalisation
-------------
  CT  : clip [-200, 300] HU  → [0, 1]
  PT  : clip [0, 20]  SUV   → [0, 1]   (standard SUV range for head/neck FDG-PET)
  Mask: values 0/1           → kept as int64

Scan alignment
--------------
CT and PT are acquired on separate devices and have different voxel grids
(different size, spacing, origin, and direction cosines).  After loading
and orientation, PT (and mask for Task 1) are resampled onto CT's exact
voxel grid using SimpleITK before the shared target-spacing resample.
This preserves physical alignment — a pure tensor interpolation would
ignore origin/direction differences and silently misalign the modalities.
"""

import json
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# ── Normalisation constants ───────────────────────────────────────────────────
_CT_MIN,  _CT_MAX  = -200.0, 300.0
_PT_MIN,  _PT_MAX  =    0.0,  20.0   # SUV


def _norm(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clip to [vmin, vmax] and rescale to [0, 1]."""
    return (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin)


def _resample_itk(
    itk_img:        sitk.Image,
    target_spacing: tuple,
    is_mask:        bool = False,
) -> sitk.Image:
    """Resample a SimpleITK image to target_spacing."""
    orig_size    = itk_img.GetSize()
    orig_spacing = itk_img.GetSpacing()
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)
    ]
    r = sitk.ResampleImageFilter()
    r.SetSize(new_size)
    r.SetOutputSpacing(target_spacing)
    r.SetOutputOrigin(itk_img.GetOrigin())
    r.SetOutputDirection(itk_img.GetDirection())
    r.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    r.SetDefaultPixelValue(0 if is_mask else -1000)
    return r.Execute(itk_img)


def _register_to_ct(
    moving:    sitk.Image,
    reference: sitk.Image,
    is_mask:   bool = False,
) -> sitk.Image:
    """
    Resample `moving` onto the physical grid of `reference` (CT).

    Copies size, spacing, origin, and direction from the reference so
    the two volumes are voxel-to-voxel aligned regardless of acquisition
    differences.  Must be called before the shared target-spacing resample.

    Parameters
    ----------
    moving    : PT or mask image to be aligned to CT space
    reference : CT image whose grid defines the output space
    is_mask   : True  -> nearest-neighbour (preserves label values)
                False -> linear interpolation
    """
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)   # copies size/spacing/origin/direction
    r.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    r.SetDefaultPixelValue(0 if is_mask else 0.0)
    return r.Execute(moving)


def _spatial_resize_2ch(
    vol: torch.Tensor,           # (D, 2, H, W)
    target_shape: tuple,
) -> torch.Tensor:
    """Resize H, W of a 2-channel volume. D treated as batch dim."""
    D = vol.shape[0]
    return TF.resize(
        vol,                     # (D, 2, H, W)
        list(target_shape),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )                            # (D, 2, H, W)


def _spatial_resize_1ch(
    vol: torch.Tensor,           # (D, H, W)
    target_shape: tuple,
    nearest: bool = False,
) -> torch.Tensor:
    interp = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
    return TF.resize(
        vol.unsqueeze(0),        # (1, D, H, W)
        list(target_shape),
        interpolation=interp,
        antialias=(not nearest),
    ).squeeze(0)                 # (D, H, W)


# ═════════════════════════════════════════════════════════════════════════════
# Task 1 — CT + PT + mask  (UNet training)
# ═════════════════════════════════════════════════════════════════════════════

class HectorTask1Dataset(Dataset):
    """
    Returns 2-channel CT+PT volumes with tumour masks for UNet training.

    Train mode : 16-slice tumour-centred crop
    Val   mode : full volume

    Return schema
    -------------
    {
      "ct_pt"  : (crop_depth|D, 2, H, W)  float32  — ch0=CT, ch1=PT
      "mask"   : (crop_depth|D, H, W)     int64
      "caseid" : str
    }
    """

    def __init__(
        self,
        task1_dir:      str,
        split_file:     str,
        target_spacing: tuple,
        target_shape:   tuple,
        mode:           str   = "train",
        p:              float = 0.8,
        crop_depth:     int   = 16,
    ):
        self.task1_dir      = task1_dir
        self.target_spacing = target_spacing
        self.target_shape   = target_shape
        self.mode           = mode
        self.p              = p
        self.crop_depth     = crop_depth

        with open(split_file) as f:
            splits = json.load(f)
        self.cases = splits["task1"][mode]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        import random
        pid = self.cases[index]
        d   = os.path.join(self.task1_dir, pid)

        ct_itk   = sitk.DICOMOrient(sitk.ReadImage(os.path.join(d, f"{pid}__CT.nii.gz")), "RAS")
        pt_itk   = sitk.DICOMOrient(sitk.ReadImage(os.path.join(d, f"{pid}__PT.nii.gz")), "RAS")
        mask_itk = sitk.DICOMOrient(sitk.ReadImage(os.path.join(d, f"{pid}.nii.gz")),     "RAS")

        # Register PT and mask onto CT grid before shared resample.
        # CT and PT come from different scanners — different sizes,
        # spacings, origins, and directions must all be reconciled.
        pt_itk   = _register_to_ct(pt_itk,   reference=ct_itk, is_mask=False)
        mask_itk = _register_to_ct(mask_itk, reference=ct_itk, is_mask=True)

        ct_itk   = _resample_itk(ct_itk,   self.target_spacing, is_mask=False)
        pt_itk   = _resample_itk(pt_itk,   self.target_spacing, is_mask=False)
        mask_itk = _resample_itk(mask_itk, self.target_spacing, is_mask=True)

        ct_arr   = _norm(sitk.GetArrayFromImage(ct_itk).astype(np.float32),   _CT_MIN, _CT_MAX)
        pt_arr   = _norm(sitk.GetArrayFromImage(pt_itk).astype(np.float32),   _PT_MIN, _PT_MAX)
        mask_arr = sitk.GetArrayFromImage(mask_itk).astype(np.int64)

        # Stack CT + PT → (D, 2, H, W)
        ct_pt = torch.from_numpy(
            np.stack([ct_arr, pt_arr], axis=1)   # (D, 2, H, W)
        )
        mask  = torch.from_numpy(mask_arr)        # (D, H, W)

        ct_pt = _spatial_resize_2ch(ct_pt, self.target_shape)
        mask  = _spatial_resize_1ch(mask.float(), self.target_shape, nearest=True).long()

        if self.mode == "train":
            ct_pt, mask = self._train_crop(ct_pt, mask, random)

        return {"ct_pt": ct_pt, "mask": mask, "caseid": pid}

    def _train_crop(self, ct_pt, mask, rng):
        import random as _random
        depth = ct_pt.shape[0]
        crop  = self.crop_depth

        nonzero = (mask != 0).any(dim=(1, 2))
        indices = torch.where(nonzero)[0]

        if len(indices) > 0 and torch.rand(()) < self.p:
            center = indices[_random.randint(0, len(indices) - 1)].item()
            z_min  = max(0, center - crop + 1)
            z_max  = min(center, depth - crop)
            z = (
                max(0, min(center, depth - crop))
                if z_max < z_min
                else _random.randint(z_min, z_max)
            )
        else:
            z = _random.randint(0, max(0, depth - crop))

        return ct_pt[z:z + crop], mask[z:z + crop]


# ═════════════════════════════════════════════════════════════════════════════
# Task 2 — CT + PT only  (survival training)
# ═════════════════════════════════════════════════════════════════════════════

class HectorTask2Dataset(Dataset):
    """
    Returns full CT+PT volumes (no mask) for survival training.

    The UNet trained on Task 1 predicts the mask at inference time —
    this dataset does not load or return any ground-truth mask.

    Survival labels
    ---------------
    event         : Relapse == 1
    survival_time : RFS (days)

    Return schema
    -------------
    {
      "ct"           : (D, H, W)  float32  — CT normalised to [0, 1]
      "pt"           : (D, H, W)  float32  — PT normalised to [0, 1]
      "caseid"       : str
      "event"        : bool tensor
      "survival_time": float32 tensor  (days)
    }

    Note: ct and pt are returned as separate tensors so that sliding_window_predict
    can receive the 2-channel stack (torch.stack([ct, pt], dim=1)) in the same
    way as during UNet training.
    """

    def __init__(
        self,
        task2_dir:      str,
        metadata_csv:   str,
        split_file:     str,
        target_spacing: tuple,
        target_shape:   tuple,
        mode:           str = "train",
    ):
        self.task2_dir      = task2_dir
        self.target_spacing = target_spacing
        self.target_shape   = target_shape
        self.mode           = mode

        with open(split_file) as f:
            splits = json.load(f)
        self.cases = splits["task2"][mode]

        df = pd.read_csv(metadata_csv)
        df = df.set_index("PatientID")
        self.metadata = df

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        pid = self.cases[index]
        d   = os.path.join(self.task2_dir, pid)

        ct_itk = sitk.DICOMOrient(sitk.ReadImage(os.path.join(d, f"{pid}__CT.nii.gz")), "RAS")
        pt_itk = sitk.DICOMOrient(sitk.ReadImage(os.path.join(d, f"{pid}__PT.nii.gz")), "RAS")

        # Register PT onto CT grid before shared resample.
        pt_itk = _register_to_ct(pt_itk, reference=ct_itk, is_mask=False)

        ct_itk = _resample_itk(ct_itk, self.target_spacing, is_mask=False)
        pt_itk = _resample_itk(pt_itk, self.target_spacing, is_mask=False)

        ct_arr = _norm(sitk.GetArrayFromImage(ct_itk).astype(np.float32), _CT_MIN, _CT_MAX)
        pt_arr = _norm(sitk.GetArrayFromImage(pt_itk).astype(np.float32), _PT_MIN, _PT_MAX)

        ct = torch.from_numpy(ct_arr)   # (D, H, W)
        pt = torch.from_numpy(pt_arr)   # (D, H, W)

        ct = _spatial_resize_1ch(ct, self.target_shape)
        pt = _spatial_resize_1ch(pt, self.target_shape)

        event, survival_time = self._get_survival(pid)

        return {
            "ct":            ct,
            "pt":            pt,
            "caseid":        pid,
            "event":         torch.tensor(event,         dtype=torch.bool),
            "survival_time": torch.tensor(survival_time, dtype=torch.float32),
        }

    def _get_survival(self, pid: str) -> tuple[bool, float]:
        row           = self.metadata.loc[pid]
        event         = bool(int(row["Relapse"]) == 1)
        survival_time = float(row["RFS"])          # already in days
        return event, survival_time