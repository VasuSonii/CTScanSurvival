# KiTS21 — Two-Phase Training Pipeline

```
kits21/
├── configs/
│   ├── unet_config.py        # All Phase 1 hyperparameters
│   └── survival_config.py    # All Phase 2 hyperparameters
├── data/
│   └── dataset.py            # KitsDataset (train crops / val full volume)
├── models/
│   ├── unet.py               # SimpleUNet3D
│   ├── omnirad.py            # OmniRadEncoder (frozen ViT wrapper)
│   └── egmdm.py              # EGMDMHead (survival mixture model)
├── losses/
│   ├── seg_loss.py           # CEDiceLoss
│   └── survival_loss.py      # EGMDMLoss (censored NLL)
├── utils/
│   ├── inference.py          # sliding_window_inference / sliding_window_predict
│   ├── logging_utils.py      # setup_logging (shared)
│   ├── metrics.py            # compute_kits_dice, concordance_index
│   └── wandb_utils.py        # debug stat accumulation / W&B helpers
├── train_unet.py             # Phase 1 entry point
└── train_survival.py         # Phase 2 entry point
```

## Phase 1 — UNet Segmentation

```bash
python -m kits21.train_unet
```

- Trains `SimpleUNet3D` on 16-slice tumour-centred crops
- Validates with full-volume sliding-window Dice (kidney / tumour / cyst, hierarchical)
- Saves `checkpoints/best_unet.pth`

## Phase 2 — Survival Analysis

```bash
python -m kits21.train_survival
```

Requires `checkpoints/best_unet.pth` from Phase 1.

Pipeline per patient:
1. Full CT volume → frozen UNet (sliding window) → predicted mask `(D, H, W)`
2. CT + mask → frozen OmniRad → per-slice embeddings `(D, 768)`
3. Mean-pool across depth → patient vector `(768,)`
4. EGMDMHead → Gaussian mixture params → EGMDMLoss (censored NLL)

Only `EGMDMHead` is trained.

## Configuration

Edit the dataclass fields in `configs/unet_config.py` or `configs/survival_config.py`.
All paths, hyperparameters, and W&B settings live there — nothing else needs touching.

## Environment

```
WANDB_API_KEY=your_key_here
```

Place in a `.env` file in the project root.