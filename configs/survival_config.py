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
    seed:            int = 123

    # ── Data paths ─────────────────────────────────────────────────────────
    root_dir:  str = "/home/sandeep/RAW_DATA/kits23/dataset"
    json_path: str = "/home/sandeep/Vasu/Kits21Model/train_test_kits23.json"

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
    device: str = "cuda:1"

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
    slice_pooling:          str = "attention"   # "mean" | "attention"
    attn_hidden_size:       int = 128      # inner dim of the attention network
    attn_dropout:           float = 0.25  # dropout inside the attention gate

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    # Capacity reduced to match dataset size (~240 patients).
    # K=10, hidden=256 gave ~1.4M params — ~6000 params/patient = overfit.
    # K=5, hidden=128 gives ~300K params — ~1250 params/patient.
    egmdm_E:           int   = 3
    egmdm_K:           int   = 5
    egmdm_hidden_size: int   = 128
    egmdm_dropout:     float = 0.3

    # ── Loss ───────────────────────────────────────────────────────────────
    lambda_div: float = 1.0    # expert spread penalty, in [0,1]
    lambda_ent: float = 0.5    # gate entropy — use all experts equally
    lambda_mix: float = 0.5    # mixture weight entropy — spread across K components

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 100
    learning_rate:       float = 1e-4
    weight_decay:        float = 5e-3  # increased from 1e-4: ~240 patients needs strong L2
    early_stop_patience: int   = 15    # reduced from 20: stop sooner after peak
    num_workers:         int   = 8

    # ── Cross-validation ──────────────────────────────────────────────────
    # When use_kfold=True, train_survival_kfold() runs n_folds experiments.
    # fold_idx is set automatically per fold — do not set manually.
    use_kfold: bool = False
    n_folds:   int  = 5
    fold_idx:  int  = -1    # -1 = not in kfold mode

    # ── Modality flags ────────────────────────────────────────────────────
    # Three valid combinations:
    #   use_imaging=True,  use_clinical=False → imaging only
    #   use_imaging=False, use_clinical=True  → clinical only
    #   use_imaging=True,  use_clinical=True  → imaging + clinical
    use_imaging:          bool  = True
    use_clinical:         bool  = False

    # ── Clinical MLP ──────────────────────────────────────────────────────
    clinical_dim:         int   = 128   # ClinicalMLP output size
    clinical_hidden_dims: list  = None  # None → [256, 128]
    clinical_dropout:     float = 0.3
    missing_threshold:    float = 0.40

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str  = "kits23-survival-folds-new"
    # notes : free-text shown on the W&B run page — describe the experiment,
    #         hypothesis, or anything you want searchable later.
    wandb_notes:  str   = ""
    # tags  : short labels for filtering/grouping in the W&B UI.
    #         e.g. ["gt_mask", "attention", "kits23"]
    wandb_tags: list = field(default_factory=lambda: ['123', 'imaging', 'attention', 'use_gt_mask', '5_fold'])
    # wandb_tags: list = field(default_factory=lambda: ['123', 'clinical', '5_fold'])

    # ── Derived (DO NOT set manually) ──────────────────────────────────────
    run_dir:                    str = field(init=False, repr=False)
    best_ckpt:                  str = field(init=False, repr=False)
    last_ckpt:                  str = field(init=False, repr=False)
    log_dir:                    str = field(init=False, repr=False)
    egmdm_input_dim:            int = field(init=False, repr=False)
    clinical_preprocessor_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.use_imaging and not self.use_clinical:
            raise ValueError("At least one of use_imaging or use_clinical must be True.")
        # Variant string keeps run dirs and W&B names distinct
        if self.use_imaging and self.use_clinical:
            self._variant = "imaging_clinical"
        elif self.use_imaging:
            self._variant = "imaging"
        else:
            self._variant = "clinical"
        if self.fold_idx >= 0:
            # k-fold: each fold writes to its own subdirectory
            self.run_dir = os.path.join(
                "runs", self.experiment_name, self._variant,
                f"kfold_{self.n_folds}", f"seed_{self.seed}",
                f"fold_{self.fold_idx}",
            )
        else:
            self.run_dir = os.path.join(
                "runs", self.experiment_name, self._variant, f"seed_{self.seed}"
            )
        self.best_ckpt = os.path.join(self.run_dir, "best.pth")
        self.last_ckpt = os.path.join(self.run_dir, "last.pth")
        self.log_dir   = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        # EGMDM input size depends on active modalities
        imaging_dim  = self.embed_dim  if self.use_imaging  else 0
        clinical_out = self.clinical_dim if self.use_clinical else 0
        self.egmdm_input_dim = imaging_dim + clinical_out
        self.clinical_preprocessor_path = os.path.join(
            self.run_dir, "clinical_preprocessor.pkl"
        )

    def to_dict(self) -> dict:
        """
        Flat dict of every config value, suitable for wandb.init(config=...).

        Derived path fields are included so W&B / logs record exactly where
        this run's outputs landed.  Tuple fields are converted to strings so
        they display cleanly in the W&B config panel.
        """
        import dataclasses
        d = dataclasses.asdict(self)
        d["target_spacing"]       = str(d["target_spacing"])
        d["target_shape"]         = str(d["target_shape"])
        d["clinical_hidden_dims"] = str(d["clinical_hidden_dims"])
        d["wandb_tags"]           = str(d["wandb_tags"])
        return d