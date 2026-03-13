"""
configs/survival_config.py
===========================
All hyperparameters for Phase 2: Survival analysis training.

Run-directory layout
--------------------
Every run is fully isolated under:

    runs/<experiment_name>/seed_<seed>/
        best.pth          ← best validation checkpoint
        last.pth          ← latest epoch checkpoint
        survival.log      ← text log
        metrics.csv       ← per-epoch CSV

unet_ckpt
---------
Points to the Phase 1 best checkpoint for the same experiment + seed.
The default assumes Phase 1 was run with the default UNetConfig
experiment_name / seed.  When launching sweeps, the launcher sets this
automatically to match the corresponding Phase 1 run.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SurvivalConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "survival_baseline"
    seed:            int = 42

    # ── Data paths ─────────────────────────────────────────────────────────
    root_dir:  str = "/home/sandeep/Vasu/kits21/kits21/data"
    json_path: str = "/home/sandeep/Vasu/KitsModel/train_test_split.json"

    # ── Frozen UNet checkpoint ─────────────────────────────────────────────
    # Set automatically by the launcher; override only for custom experiments.
    unet_ckpt: str = os.path.join(
        "runs", "unet_baseline", "seed_42", "best.pth"
    )

    # ── Resume (optional — full path to a .pth file) ───────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: Tuple[float, float, float] = (0.78, 0.78, 3.0)
    target_shape:   Tuple[int, int]            = (256, 256)

    # ── UNet architecture (must match the checkpoint) ──────────────────────
    num_classes:        int  = 4
    unet_base_channels: int  = 32
    unet_trilinear:     bool = True

    # ── Sliding-window (UNet inference) ────────────────────────────────────
    sw_window: int = 16
    sw_stride: int = 8

    # ── Device ────────────────────────────────────────────────────────────-
    # Override this to pin a run to a specific GPU (e.g. 'cuda:1').
    device: str = "cuda:2"

    # ── Mask source ────────────────────────────────────────────────────────
    # True  → use the ground-truth segmentation mask from the dataset.
    #         Useful for upper-bound experiments: removes UNet error from the
    #         survival pipeline so you can isolate EGMDM performance.
    # False → run the frozen UNet in sliding-window mode to predict the mask.
    #         This is the realistic inference-time setting.
    use_gt_mask: bool = True

    # ── OmniRad (frozen) ───────────────────────────────────────────────────
    embed_dim:  int = 768   # OmniRad-base ViT output dimension
    omni_batch: int = 16    # slices per OmniRad forward pass — tune to VRAM

    # ── Slice pooling ──────────────────────────────────────────────────────
    # How per-slice OmniRad embeddings are aggregated into a single patient
    # vector before the EGMDM head.
    #   "mean"      — unweighted mean across depth (no extra parameters)
    #   "attention" — gated attention pooling (Ilse et al. 2018); trains a
    #                 small attention network on top of frozen OmniRad
    slice_pooling:          str = "mean"   # "mean" | "attention"
    attn_hidden_size:       int = 128      # inner dim of the attention network
    attn_dropout:           float = 0.25  # dropout inside the attention gate

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
    wandb_project: str = "kits21-survival"

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
        this run's outputs landed.  Tuple fields are converted to strings so
        they display cleanly in the W&B config panel.
        """
        import dataclasses
        d = dataclasses.asdict(self)
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        return d