# KiTS23 — Two-Phase Survival Analysis Pipeline

A deep learning pipeline for kidney tumour survival prediction using the KiTS23 dataset. Phase 1 trains a 3D UNet for segmentation; Phase 2 uses frozen OmniRad ViT embeddings and an Ensemble of Gaussian Mixture Density Models (EGMDM) for censored survival analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Phase 1 — UNet Segmentation](#phase-1--unet-segmentation)
- [Phase 2 — Survival Analysis](#phase-2--survival-analysis)
- [HECKTOR Extension](#hecktor-extension)
- [Models](#models)
- [Configuration](#configuration)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Modality Variants](#modality-variants)
- [Logging and Checkpoints](#logging-and-checkpoints)
- [Environment Variables](#environment-variables)

---

## Overview

```
CT Volume
    │
    ▼
┌─────────────┐          Phase 1 (frozen at Phase 2)
│  SimpleUNet │  ──────► Predicted Mask  (D, H, W)
│    3D       │
└─────────────┘
         │
         ▼
┌──────────────────┐     Phase 2 — frozen
│  OmniRad ViT     │  ──► Per-slice Embeddings  (D, 768)
│  (frozen)        │
└──────────────────┘
         │
         ▼  GatedAttentionPooling (trainable) or mean
         │
    Patient Vector  (768,)
         │
         ▼
┌──────────────────┐     Phase 2 — trainable
│   EGMDMHead      │  ──► Gaussian Mixture Params
│  (E experts,     │
│   K components)  │
└──────────────────┘
         │
         ▼
   EGMDMLoss (censored NLL + regularisation)
   Optimises: C-index on validation set
```

The pipeline supports three modality combinations at Phase 2:

| Mode | `use_imaging` | `use_clinical` |
|---|---|---|
| Imaging only | ✅ | ❌ |
| Clinical only | ❌ | ✅ |
| Imaging + Clinical | ✅ | ✅ |

---

## Project Structure

```
kits21/
├── configs/
│   ├── unet_config.py              # Phase 1 hyperparameters
│   ├── survival_config.py          # Phase 2 hyperparameters (KiTS)
│   └── hector_survival_config.py   # Phase 2 hyperparameters (HECKTOR)
│
├── data/
│   ├── dataset.py                  # KitsDataset — train crops / val full volume
│   ├── dataset_hector.py           # HectorTask2Dataset (CT + PET)
│   └── clinical_preprocessor.py   # ClinicalPreprocessor / ClinicalPreprocessorHector
│
├── models/
│   ├── unet.py                     # SimpleUNet3D
│   ├── omnirad.py                  # OmniRadEncoder (frozen ViT) + GatedAttentionPooling
│   ├── egmdm.py                    # EGMDMHead (survival mixture model)
│   └── clinical_mlp.py             # ClinicalMLP (clinical feature encoder)
│
├── losses/
│   ├── seg_loss.py                 # CEDiceLoss (Phase 1)
│   └── survival_loss.py            # EGMDMLoss — censored NLL + regularisation
│
├── utils/
│   ├── inference.py                # sliding_window_predict
│   ├── kfold.py                    # make_kfold_splits, load_events_from_metadata
│   ├── logging_utils.py            # setup_logging, log_config
│   ├── metrics.py                  # compute_kits_dice, concordance_index
│   ├── seed.py                     # set_seed (reproducibility)
│   └── wandb_utils.py              # W&B helper utilities
│
├── train_unet.py                   # Phase 1 entry point
├── train_survival.py               # Phase 2 entry point (KiTS)
└── train_survival_hector.py        # Phase 2 entry point (HECKTOR)
```

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA.

```bash
pip install torch torchvision timm wandb python-dotenv tqdm scikit-learn pandas numpy
```

Create a `.env` file in the project root:

```
WANDB_API_KEY=your_key_here
```

Prepare the KiTS23 dataset and update the paths in `configs/survival_config.py`:

```python
root_dir  = "/path/to/kits23/dataset"
json_path = "/path/to/train_test_kits23.json"
```

---

## Phase 1 — UNet Segmentation

```bash
python -m kits21.train_unet
```

Trains `SimpleUNet3D` to segment kidney, tumour, and cyst from 3D CT volumes.

**Training details:**
- Input: 16-slice tumour-centred random crops `(16, 256, 256)`
- Loss: Combined CE + Dice (`CEDiceLoss`)
- Validation: full-volume sliding-window inference, hierarchical Dice (kidney / tumour / cyst)
- Output: `runs/<experiment_name>/seed_<seed>/best.pth`

All hyperparameters (architecture, data paths, training schedule) are in `configs/unet_config.py`.

---

## Phase 2 — Survival Analysis

```bash
# Single run
python -m kits21.train_survival

# K-fold cross-validation
python -m kits21.train_survival --kfold
```

Requires a Phase 1 checkpoint. Set `unet_ckpt` in `configs/survival_config.py` to point to it.

**Per-patient pipeline:**

1. **Mask prediction** — frozen UNet runs sliding-window inference over the full CT volume → `(D, H, W)` segmentation mask.  
   *(Or use ground-truth mask with `use_gt_mask=True` for upper-bound experiments.)*

2. **OmniRad encoding** — each axial slice is assembled as a 3-channel tensor `[CT, mask_norm, CT]` and passed through the frozen OmniRad ViT → `(D, 768)` per-slice embeddings.

3. **Slice pooling** — `(D, 768)` → `(768,)` patient vector via either:
   - `"mean"` — unweighted mean, no extra parameters
   - `"attention"` — gated attention pooling (Ilse et al. 2018), jointly trained with the survival head

4. **Clinical fusion** *(optional)* — clinical features are encoded by a lightweight `ClinicalMLP` to `(clinical_dim,)` and L2-normalised before concatenation with the imaging embedding.

5. **EGMDM head** — patient vector → Gaussian mixture parameters → censored NLL loss.

**What is trained:** only `EGMDMHead`, `GatedAttentionPooling` (if attention), and `ClinicalMLP` (if clinical). OmniRad and UNet are fully frozen.

---

## HECKTOR Extension

Head-and-neck cancer survival analysis using CT + PET from the HECKTOR 2022 challenge (Task 2).

```bash
# Single run
python train_survival_hector.py

# K-fold CV
python train_survival_hector.py --kfold
```

Key differences from KiTS:

| | KiTS | HECKTOR |
|---|---|---|
| Modalities | CT only | CT + PET |
| OmniRad channels | `[CT, mask_norm, CT]` | `[CT, PT, mask_norm]` |
| UNet input channels | 1 | 2 (CT + PT) |
| Segmentation classes | 4 (bg, kidney, tumour, cyst) | 2 (bg, tumour) |
| Tumour crop | ❌ | ✅ (tumour-centred depth crop at training) |
| Gradient accumulation | ❌ | ✅ (`accumulation_steps`) |

Config: `configs/hector_survival_config.py`

---

## Models

### SimpleUNet3D (`models/unet.py`)
Standard encoder-decoder with skip connections. Supports trilinear or transposed-convolution upsampling. Configurable base channel width and input channels.

### OmniRadEncoder (`models/omnirad.py`)
Wrapper around the pretrained `OmniRad-base` ViT from HuggingFace Hub. Always frozen. Encodes individual slices or full volumes. Supports CT-only (`encode_volume`) and CT+PET (`encode_volume_ct_pt`) modes.

### GatedAttentionPooling (`models/omnirad.py`)
Aggregates variable-length per-slice embeddings into a single patient vector using the gated attention mechanism of Ilse et al. (ICML 2018):

```
a_i = softmax( w^T (tanh(V h_i) ⊙ sigmoid(U h_i)) )
z   = Σ_i a_i h_i
```

Trainable. Attention weights are logged to W&B for interpretability.

### EGMDMHead (`models/egmdm.py`)
Ensemble of Gaussian Mixture Density Models. Each of `E` expert MLPs produces `K` Gaussian components over log-transformed survival time. A gating network combines experts, and the resulting mixture models the survival distribution.

Key outputs:
- `params` — `{'w', 'mu', 'sigma'}` mixture parameters
- `reg_losses` — `{'L_div', 'L_ent', 'L_mix'}` regularisation terms

**Regularisation:**
- `L_div` — expert diversity: penalises collapsed mean centres across experts
- `L_ent` — gate entropy: encourages all experts to be used
- `L_mix` — mixture entropy: prevents the model from collapsing to a narrow distribution per patient

### ClinicalMLP (`models/clinical_mlp.py`)
Lightweight MLP with LayerNorm (not BatchNorm — compatible with per-patient batch size of 1). Maps preprocessed clinical features to a fixed-size embedding.

### EGMDMLoss (`losses/survival_loss.py`)
Handles right-censored survival data:
- **Event (event=1):** maximise `log p(t)`
- **Censored (event=0):** maximise `log S(t) = log(1 − CDF(t))`

Total loss: `NLL + λ_div · L_div + λ_ent · L_ent + λ_mix · L_mix`

---

## Configuration

All hyperparameters are dataclass fields — edit them directly in the config files, no CLI parsing needed.

### Key fields in `SurvivalConfig`

| Field | Default | Description |
|---|---|---|
| `use_gt_mask` | `True` | Use ground-truth mask (upper bound) vs predicted mask (realistic) |
| `slice_pooling` | `"attention"` | `"mean"` or `"attention"` |
| `use_imaging` | `True` | Include imaging pathway |
| `use_clinical` | `False` | Include clinical MLP pathway |
| `egmdm_E` | `3` | Number of EGMDM experts |
| `egmdm_K` | `5` | Gaussian components per expert |
| `lambda_div` | `1.0` | Expert diversity regularisation weight |
| `lambda_ent` | `0.5` | Gate entropy regularisation weight |
| `lambda_mix` | `0.5` | Mixture entropy regularisation weight |
| `early_stop_patience` | `15` | Epochs without C-index improvement before stopping |
| `device` | `"cuda:1"` | Override to pin to a specific GPU |

### Run directory layout

Each run writes exclusively to its own directory, determined by experiment name, variant, seed, and fold:

```
runs/
└── <experiment_name>/
    └── <variant>/              # imaging | clinical | imaging_clinical
        ├── seed_<seed>/        # single-run mode
        │   ├── best.pth
        │   ├── last.pth
        │   ├── survival.log
        │   ├── metrics.csv
        │   └── clinical_preprocessor.pkl
        └── kfold_<k>/
            └── seed_<seed>/
                └── fold_<i>/
                    ├── best.pth
                    └── ...
```

---

## K-Fold Cross-Validation

```bash
python -m kits21.train_survival --kfold       # KiTS
python train_survival_hector.py --kfold        # HECKTOR
```

Set `use_kfold=True` and `n_folds=5` in the config (or pass `--kfold` at runtime).

- Stratified splits by event label (preserves event rate per fold)
- Each fold trains a fresh model and logs its own W&B run in the same group
- A summary W&B run is created after all folds with mean ± std C-index
- Best checkpoint path per fold is returned for downstream ensemble evaluation

---

## Modality Variants

Three valid configurations, selected via `use_imaging` and `use_clinical`:

**Imaging only** — upper bound for imaging contribution:
```python
use_imaging  = True
use_clinical = False
```

**Clinical only** — baseline using structured patient data:
```python
use_imaging  = False
use_clinical = True
```

**Imaging + Clinical** — full multimodal fusion:
```python
use_imaging  = True
use_clinical = True
```

Both embeddings are L2-normalised to unit norm before concatenation so neither modality dominates by magnitude. The `egmdm_input_dim` is computed automatically from the active modalities.

---

## Logging and Checkpoints

**W&B metrics logged per epoch:**

| Metric | Description |
|---|---|
| `val/cindex` | Harrell's C-index on validation set |
| `val/sigma_mean` / `val/sigma_min` | EGMDM spread diagnostics — low values indicate memorisation |
| `val/mixture_entropy` | Entropy of mixture weights — low = overconfident, poor discrimination |
| `val/attn_entropy` | Attention weight entropy — low = attention collapsed to few slices |
| `val/attn_max_weight` | Max attention weight per patient — near 1 = saturated |
| `train/grad_egmdm` / `train/grad_pooling` / `train/grad_clinical` | Per-module gradient norms — tells you which modules are learning |
| `train/img_emb_norm` / `train/clin_emb_norm` | Embedding magnitude before normalisation |

**Checkpoints:**
- `best.pth` — saved whenever validation C-index improves
- `last.pth` — saved every epoch (for resuming)

**Resuming a run:**
```python
cfg.resume_path = "runs/my_exp/imaging/seed_123/last.pth"
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WANDB_API_KEY` | ✅ | Weights & Biases API key |

Place in `.env` in the project root — loaded automatically via `python-dotenv`.

---

## Citation

If you use OmniRad embeddings, please cite the original model:

```
OmniRad-base: Snarcy/OmniRad-base (HuggingFace Hub)
```

Gated attention pooling:
```
Ilse, M., Tomczak, J. M., & Welling, M. (2018).
Attention-based Deep Multiple Instance Learning. ICML 2018.
https://arxiv.org/abs/1802.04712
```
