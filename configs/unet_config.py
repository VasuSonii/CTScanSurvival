"""
configs/unet_config.py
======================
All hyperparameters for Phase 1: UNet segmentation training.

Run-directory layout
--------------------
Every run is fully isolated under:

    runs/<experiment_name>/seed_<seed>/
        best.pth          ← best validation checkpoint
        last.pth          ← latest epoch checkpoint
        train.log         ← text log
        metrics.csv       ← per-epoch CSV

The only fields callers should ever set are the ones listed below;
run_dir / best_ckpt / last_ckpt / log_dir are all derived automatically
in __post_init__ and should NOT be overridden.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class UNetConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "unet_baseline"
    seed:            int = 42

    # ── Data paths ─────────────────────────────────────────────────────────
    root_dir:  str = "/home/sandeep/Vasu/kits21/kits21/data"
    json_path: str = "/home/sandeep/Vasu/KitsModel/train_test_split.json"

    # ── Resume (optional — full path to a .pth file) ───────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: Tuple[float, float, float] = (0.78, 0.78, 3.0)
    target_shape:   Tuple[int, int]            = (256, 256)
    tumour_crop_p:  float = 0.8

    # ── Model ──────────────────────────────────────────────────────────────
    num_classes:   int  = 4
    base_channels: int  = 32
    trilinear:     bool = True

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 100
    batch_size:          int   = 1
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-4
    accumulation_steps:  int   = 4
    early_stop_patience: int   = 35
    num_workers:         int   = 16

    # ── Loss ───────────────────────────────────────────────────────────────
    ce_weight:     float      = 0.1
    class_weights: List[float] = field(default_factory=lambda: [0.1, 3.0, 5.0, 7.0])

    # ── Sliding-window (validation) ────────────────────────────────────────
    sw_window: int = 16
    sw_stride: int = 8

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str = "kits21-segmentation"

    # ── Derived paths (DO NOT set manually) ───────────────────────────────
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
        """
        Flat dict of every config value, suitable for wandb.init(config=...).

        Derived path fields are included so W&B / logs record exactly where
        this run's outputs landed.  List fields are converted to strings so
        they display cleanly in the W&B config panel.
        """
        import dataclasses
        d = dataclasses.asdict(self)
        # Convert lists to strings for clean W&B display
        d["class_weights"] = str(d["class_weights"])
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        return d