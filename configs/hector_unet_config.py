"""
configs/hector_unet_config.py
==============================
Hyperparameters for training a 2-channel (CT+PT) UNet on HECKTOR Task 1.

The UNet trained here is later used frozen in train_survival_hector.py
to predict tumour masks for Task 2 patients.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HectorUNetConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "hector_unet"
    seed:            int = 42

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:1"

    # ── Data paths ─────────────────────────────────────────────────────────
    task1_dir:  str = "/home/sandeep/RAW_DATA/HECKTOR2025/HECKTOR_2025_Training_Data/Task 1"
    split_file: str = "/home/sandeep/RAW_DATA/HECKTOR2025/HECKTOR_2025_Training_Data/dataset_split_fixed.json"

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: tuple = (1.0, 1.0, 3.0)
    target_shape:   tuple = (256, 256)
    tumour_crop_p:  float = 0.8
    crop_depth:     int   = 16

    # ── Model ──────────────────────────────────────────────────────────────
    in_channels:   int  = 2          # CT + PT
    num_classes:   int  = 3          # 0=background, 1=tumour, 2=lymph node
    base_channels: int  = 32
    trilinear:     bool = True

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 200
    batch_size:          int   = 16
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-4
    accumulation_steps:  int   = 4
    early_stop_patience: int   = 30
    num_workers:         int   = 20

    # ── Loss ───────────────────────────────────────────────────────────────
    # loss_type='dice'    → CEDiceLoss    (equal FP/FN penalty)
    # loss_type='tversky' → CETverskyLoss (FN penalised >> FP, broader mask)
    loss_type:     str   = "tversky"
    tversky_alpha: float = 0.2   # FP weight — low: tolerate over-prediction
    tversky_beta:  float = 0.8   # FN weight — high: penalise missing tumour

    ce_weight:     float = 0.5
    # Weights for [background, tumour, lymph node].
    # Background is downweighted; lymph nodes are typically smaller than
    # the primary tumour so they get a higher weight to counter class imbalance.
    class_weights: list  = field(default_factory=lambda: [0.1, 0.4, 0.5])

    # ── Sliding window (val) ───────────────────────────────────────────────
    sw_window: tuple = (16, 256, 256)
    sw_stride: tuple = (8,  128, 128)

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str  = "hector-unet-new"
    # Use tags to filter runs by loss type, e.g. ["tversky", "alpha0.2"]
    # loss_type is auto-added as a tag so you can always filter by it.
    wandb_notes:   str  = ""
    wandb_tags: list = field(default_factory=lambda: ['unet', 'tversky', 'alpha0.2', 'beta0.8', '42'])

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

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["class_weights"]  = str(d["class_weights"])
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        d["sw_window"]      = str(d["sw_window"])
        d["sw_stride"]      = str(d["sw_stride"])
        d["wandb_tags"]     = str(d["wandb_tags"])
        return d