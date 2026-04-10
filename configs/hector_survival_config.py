"""
configs/hector_survival_config.py
===================================
Hyperparameters for HECKTOR Task 2 survival training.

Three modality variants (controlled by use_imaging / use_clinical):
  imaging only          → use_imaging=True,  use_clinical=False
  clinical only         → use_imaging=False, use_clinical=True
  imaging + clinical    → use_imaging=True,  use_clinical=True

Imaging pipeline:
  HectorTask2Dataset (CT + PT)
    → 2-channel UNet [frozen, Task 1] predicts mask
    → OmniRad encodes [CT, PT, mask] per slice [frozen]
    → GatedAttentionPooling [trainable] → (768,)

Clinical pipeline (very limited — ~8-12 features after encoding):
  Age, Gender, Tobacco, Alcohol, Treatment, M-stage
    → ClinicalPreprocessorHector → ClinicalMLP [trainable] → (32,)
    Note: clinical_dim=32 (not 128) — input is only ~10 features.

No ground-truth mask for Task 2 — UNet always used.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HectorSurvivalConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "hector_survival-folds"
    seed:            int = 42

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:2"

    # ── Data paths ─────────────────────────────────────────────────────────
    task2_dir:    str = "/home/sandeep/RAW_DATA/HECKTOR2025/HECKTOR_2025_Training_Data/Task 2"
    metadata_csv: str = "/home/sandeep/RAW_DATA/HECKTOR2025/HECKTOR_2025_Training_Data/Task 2/HECKTOR_2025_Training_Task_2.csv"
    split_file:   str = "/home/sandeep/RAW_DATA/HECKTOR2025/HECKTOR_2025_Training_Data/dataset_split_fixed.json"

    # ── UNet checkpoint (trained on Task 1) ───────────────────────────────
    unet_ckpt: Optional[str] = None  # derived in __post_init__ from seed

    # ── Resume ────────────────────────────────────────────────────────────
    resume_path: Optional[str] = None

    # ── Cross-validation ──────────────────────────────────────────────────
    use_kfold: bool = False
    n_folds:   int  = 5
    fold_idx:  int  = -1   # -1 = not in kfold mode; set automatically per fold

    # ── Modality flags ────────────────────────────────────────────────────
    # imaging only        : use_imaging=True,  use_clinical=False
    # clinical only       : use_imaging=False, use_clinical=True
    # imaging + clinical  : use_imaging=True,  use_clinical=True
    use_imaging:  bool = True
    use_clinical: bool = True

    # ── Clinical MLP ──────────────────────────────────────────────────────
    # HECKTOR has ~8-12 clinical features after encoding.
    # clinical_dim=32 is appropriate — 128 would overparameterise this input.
    clinical_dim:         int   = 32
    clinical_hidden_dims: list  = None   # None → [64, 32]
    clinical_dropout:     float = 0.3
    missing_threshold:    float = 0.40

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: tuple = (1.0, 1.0, 3.0)
    target_shape:   tuple = (256, 256)

    # ── UNet architecture (must match Task 1 checkpoint) ──────────────────
    in_channels:        int  = 2     # CT + PT
    num_classes:        int  = 3     # 0=background, 1=tumour, 2=lymph node
    unet_base_channels: int  = 32
    unet_trilinear:     bool = True

    # ── Training crop ─────────────────────────────────────────────────────
    crop_depth:         int   = 16
    tumour_crop_p:      float = 0.8
    accumulation_steps: int   = 4

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
    egmdm_K:           int   = 5
    egmdm_hidden_size: int   = 128
    egmdm_dropout:     float = 0.3

    # ── Loss ───────────────────────────────────────────────────────────────
    lambda_div: float = 1.0
    lambda_ent: float = 0.5
    lambda_mix: float = 0.5

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 100
    learning_rate:       float = 1e-4
    weight_decay:        float = 5e-3
    early_stop_patience: int   = 15
    num_workers:         int   = 8

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str  = "hector-survival-folds"
    wandb_notes:   str  = ""
    wandb_tags: list = field(default_factory=lambda: ['42', 'imaging', 'attention', '5_fold', 'clinical'])

    # ── Derived (DO NOT set manually) ──────────────────────────────────────
    run_dir:                    str = field(init=False, repr=False)
    best_ckpt:                  str = field(init=False, repr=False)
    last_ckpt:                  str = field(init=False, repr=False)
    log_dir:                    str = field(init=False, repr=False)
    egmdm_input_dim:            int = field(init=False, repr=False)
    clinical_preprocessor_path: str = field(init=False, repr=False)
    _variant:                   str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.use_imaging and not self.use_clinical:
            raise ValueError("At least one of use_imaging or use_clinical must be True.")

        if self.use_imaging and self.use_clinical:
            self._variant = "imaging_clinical"
        elif self.use_imaging:
            self._variant = "imaging"
        else:
            self._variant = "clinical"

        if self.fold_idx >= 0:
            self.run_dir = os.path.join(
                "runs", self.experiment_name, self._variant,
                f"kfold_{self.n_folds}", f"seed_{self.seed}", f"fold_{self.fold_idx}",
            )
        else:
            self.run_dir = os.path.join(
                "runs", self.experiment_name, self._variant, f"seed_{self.seed}"
            )

        self.best_ckpt = os.path.join(self.run_dir, "best.pth")
        self.last_ckpt = os.path.join(self.run_dir, "last.pth")
        self.log_dir   = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        imaging_dim  = self.embed_dim    if self.use_imaging  else 0
        clinical_out = self.clinical_dim if self.use_clinical else 0
        self.egmdm_input_dim = imaging_dim + clinical_out

        self.clinical_preprocessor_path = os.path.join(
            self.run_dir, "clinical_preprocessor.pkl"
        )

        if self.unet_ckpt is None:
            self.unet_ckpt = os.path.join(
                "runs", "hector_unet", f"seed_{self.seed}", "best.pth"
            )

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["target_spacing"]       = str(d["target_spacing"])
        d["target_shape"]         = str(d["target_shape"])
        d["sw_window"]            = str(d["sw_window"])
        d["sw_stride"]            = str(d["sw_stride"])
        d["clinical_hidden_dims"] = str(d["clinical_hidden_dims"])
        d["wandb_tags"]           = str(d["wandb_tags"])
        return d