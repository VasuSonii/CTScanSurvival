"""
configs/comorbidity_config.py
==============================
Hyperparameters for probing whether OmniRad image features can predict
comorbidities from CT scans alone.

Pipeline
--------
  KitsDataset (CT + GT mask)
    → OmniRad.encode_volume()     [frozen]     → (D, 768)
    → GatedAttentionPooling       [trainable]  → (768,)
    → ComorbidityClassifier       [trainable]  → (5,) logits
    → BCEWithLogitsLoss (per label, pos-weighted)

Goal: measure how much patient health state is encoded in CT imaging.
If AUC is high → OmniRad features encode comorbidity signal → explains
why imaging alone is competitive with imaging+clinical in survival task.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComorbidityConfig:
    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "comorbidity_probe"
    seed:            int = 42

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"

    # ── Data paths ─────────────────────────────────────────────────────────
    root_dir:  str = "/home/sandeep/RAW_DATA/kits23/dataset"
    json_path: str = "/home/sandeep/Vasu/Kits21Model/train_test_kits23.json"

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_path: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    target_spacing: tuple = (0.78, 0.78, 3.0)
    target_shape:   tuple = (256, 256)
    use_gt_mask:    bool  = True   # GT mask used as OmniRad input channel

    # ── OmniRad (always frozen) ────────────────────────────────────────────
    embed_dim:  int = 768
    omni_batch: int = 16

    # ── Slice pooling (trainable) ──────────────────────────────────────────
    slice_pooling:    str   = "attention"
    attn_hidden_size: int   = 128
    attn_dropout:     float = 0.25

    # ── Classifier head (trainable) ────────────────────────────────────────
    classifier_hidden_dim: int   = 128
    classifier_dropout:    float = 0.3

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs:          int   = 100
    learning_rate:       float = 1e-4
    weight_decay:        float = 5e-3
    early_stop_patience: int   = 15
    num_workers:         int   = 8

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_project: str  = "kits23-comorbidity-probe"
    wandb_notes:   str  = ""
    wandb_tags:    list = None

    # ── Derived paths ───────────────────────────────────────────────────────
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
        d["target_spacing"] = str(d["target_spacing"])
        d["target_shape"]   = str(d["target_shape"])
        d["wandb_tags"]     = str(d["wandb_tags"])
        return d