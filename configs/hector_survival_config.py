"""
configs/hector_survival_config.py
===================================
Hyperparameters for HECKTOR Task 2 survival training.

Pipeline
--------
  HectorTask2Dataset (CT + PT)
    → 2-channel UNet predicts tumour mask  [frozen, trained on Task 1]
    → OmniRad encodes [CT, PT, mask] per slice  [frozen]
    → GatedAttentionPooling  [trainable]
    → EGMDMHead  [trainable]
    → RFS survival prediction

No ground-truth mask is available for Task 2 — use_gt_mask is not present.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HectorSurvivalConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "hector_survival_tversky"
    seed:            int = 98

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:1"

    # ── Data paths ─────────────────────────────────────────────────────────
    task2_dir:    str = "/home/sandeep/HECKTOR2025/HECKTOR_2025_Training_Data/Task 2"
    metadata_csv: str = "/home/sandeep/HECKTOR2025/HECKTOR_2025_Training_Data/Task 2/HECKTOR_2025_Training_Task_2.csv"
    split_file:   str = "/home/sandeep/HECKTOR2025/HECKTOR_2025_Training_Data/dataset_split_fixed.json"

    # ── UNet checkpoint (trained on Task 1) ───────────────────────────────
    unet_ckpt: Optional[str] = None  # derived in __post_init__ from seed

    # ── Resume survival training ───────────────────────────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: tuple = (1.0, 1.0, 3.0)
    target_shape:   tuple = (256, 256)

    # ── UNet architecture (must match the trained checkpoint) ─────────────
    in_channels:       int  = 2     # CT + PT
    num_classes:       int  = 3     # 0=background, 1=tumour, 2=lymph node
    unet_base_channels:int  = 32
    unet_trilinear:    bool = True

    # ── Training crop ─────────────────────────────────────────────────────
    # During training, UNet runs on the full volume to find tumour slices,
    # then only crop_depth slices are passed to OmniRad.  This makes every
    # training sample a fixed-depth tensor so gradient accumulation can
    # simulate a larger effective batch size.
    # Validation always uses the full volume (DataLoader batch_size=1).
    crop_depth:         int   = 16
    tumour_crop_p:      float = 0.8   # prob of tumour-centred vs random crop
    accumulation_steps: int   = 4     # effective batch = accumulation_steps

    # ── Sliding window for UNet inference ─────────────────────────────────
    sw_window: tuple = (16, 256, 256)
    sw_stride: tuple = (8,  128, 128)

    # ── OmniRad ────────────────────────────────────────────────────────────
    embed_dim:  int = 768
    omni_batch: int = 16

    # ── Slice pooling ──────────────────────────────────────────────────────
    slice_pooling:    str   = "attention"
    attn_hidden_size: int   = 128
    attn_dropout:     float = 0.25

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm_E:           int   = 3
    egmdm_K:           int   = 10
    egmdm_hidden_size: int   = 256
    egmdm_dropout:     float = 0.1

    # ── Loss ───────────────────────────────────────────────────────────────
    lambda_div: float = 0.1
    lambda_ent: float = 0.01

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 100
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-4
    early_stop_patience: int   = 20
    num_workers:         int   = 8

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str = "hector-survival"

    # ── Derived paths (DO NOT set manually) ────────────────────────────────
    run_dir:   str = field(init=False, repr=False)
    best_ckpt: str = field(init=False, repr=False)
    last_ckpt: str = field(init=False, repr=False)
    log_dir:   str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_dir   = os.path.join("runs", self.experiment_name, f"seed_{self.seed}")
        self.best_ckpt = os.path.join(self.run_dir, "best.pth")
        self.last_ckpt = os.path.join(self.run_dir, "last.pth")
        self.log_dir   = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        # Default: each survival seed loads its own seed-matched UNet.
        # Override unet_ckpt explicitly to share a single UNet across seeds.
        if self.unet_ckpt is None:
            self.unet_ckpt = os.path.join(
                "runs", "hector_unet_tversky", f"seed_{self.seed}", "best.pth"
            )

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        d["sw_window"]      = str(d["sw_window"])
        d["sw_stride"]      = str(d["sw_stride"])
        return d