"""
configs/survival_rgb_config.py
================================
Hyperparameters for the RGB-OmniRad survival experiment.

Differences from SurvivalConfig
---------------------------------
  - No UNet / sliding-window / mask fields — the encoder takes raw
    HU-windowed CT directly, no segmentation step needed.
  - Adds `hu_windows`: three (hu_min, hu_max) pairs defining the R, G, B
    channels built from the CT HU values.

Run-directory layout (same as other experiments)
-------------------------------------------------
    runs/<experiment_name>/seed_<seed>/
        best.pth
        last.pth
        survival_rgb.log
        metrics.csv
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SurvivalRGBConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "survival_rgb_baseline"
    seed:            int = 123

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:3"

    # ── Data paths ─────────────────────────────────────────────────────────
    root_dir:  str = "/home/sandeep/Vasu/kits21/kits21/data"
    json_path: str = "/home/sandeep/Vasu/KitsModel/train_test_split.json"

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: Tuple[float, float, float] = (0.78, 0.78, 3.0)
    target_shape:   Tuple[int, int]            = (256, 256)

    # ── HU windows (R, G, B) ───────────────────────────────────────────────
    # Each tuple is (hu_min, hu_max).  The CT is NOT globally clipped;
    # each channel is independently windowed and rescaled to [0, 1].
    #
    # Defaults (abdominal / renal oncology):
    #   R : soft tissue       [-160,  240]  organ edges, fat, muscle
    #   G : corticomedullary  [-100,  400]  vascular enhancement, tumour blush
    #   B : bone / structure  [-500, 1300]  skeletal context, calcifications
    hu_windows: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (-160.0,  240.0),
            (-100.0,  400.0),
            (-500.0, 1300.0),
        ]
    )

    # ── OmniRad (frozen) ───────────────────────────────────────────────────
    embed_dim:  int = 768
    omni_batch: int = 16    # slices per OmniRad forward pass — tune to VRAM

    # ── Slice pooling ──────────────────────────────────────────────────────
    # "mean"      — unweighted mean across depth
    # "attention" — gated attention pooling (Ilse et al. 2018)
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
    num_workers:         int   = 16

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str = "kits21-survival-rgb"

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
        """Flat dict suitable for wandb.init(config=...) and log_config()."""
        import dataclasses
        d = dataclasses.asdict(self)
        # Stringify compound types for clean W&B display
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        d["hu_windows"]     = str(d["hu_windows"])
        return d