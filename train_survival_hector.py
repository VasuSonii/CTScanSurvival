"""
train_survival_hector.py — HECKTOR Task 2 survival training
============================================================
Run single experiment:
    python train_survival_hector.py

Run k-fold CV:
    python train_survival_hector.py --kfold

Three modality variants (set in HectorSurvivalConfig):
  imaging only       : use_imaging=True,  use_clinical=False
  clinical only      : use_imaging=False, use_clinical=True
  imaging+clinical   : use_imaging=True,  use_clinical=True

Gradient-correct pooling
-------------------------
OmniRad encoding is @no_grad (frozen).  GatedAttentionPooling runs
inside the training loop with an active grad context so gradients
flow correctly back through pooling parameters.  The embedding stays
on GPU until after backward() — moving to CPU before backward breaks
the computation graph and causes pooling to receive zero gradients.

Imaging/clinical norm balance
------------------------------
Both embeddings are L2-normalised to unit norm before concatenation
so neither modality dominates by magnitude.
"""

import csv
import dataclasses
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.hector_survival_config       import HectorSurvivalConfig
from data.clinical_preprocessor           import ClinicalPreprocessorHector
from data.dataset_hector                  import HectorTask2Dataset
from losses.survival_loss                 import EGMDMLoss
from models.clinical_mlp                 import ClinicalMLP
from models.egmdm                        import EGMDMHead
from models.omnirad                      import GatedAttentionPooling, OmniRadEncoder
from models.unet                         import SimpleUNet3D
from utils.inference                     import sliding_window_predict
from utils.logging_utils                 import log_config, setup_logging
from utils.metrics                       import concordance_index
from utils.seed                          import set_seed
from utils.kfold                         import make_kfold_splits

logger = logging.getLogger(__name__)


# ─── Crop helper ─────────────────────────────────────────────────────────────

def _tumour_crop(mask: torch.Tensor, crop_depth: int, p: float) -> tuple[int, int]:
    D             = mask.shape[0]
    tumour_slices = (mask > 0).any(dim=(1, 2))
    indices       = torch.where(tumour_slices)[0]
    if len(indices) > 0 and torch.rand(()).item() < p:
        center = indices[random.randint(0, len(indices) - 1)].item()
        z_min  = max(0, center - crop_depth + 1)
        z_max  = min(center, D - crop_depth)
        z      = (
            max(0, min(center, D - crop_depth))
            if z_max < z_min else random.randint(z_min, z_max)
        )
    else:
        z = random.randint(0, max(0, D - crop_depth))
    return z, z + crop_depth


# ─── Embedding helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def get_slice_embeddings_hector(
    ct:      torch.Tensor,        # (D, H, W) or (D_crop, H, W)
    pt:      torch.Tensor,
    mask:    torch.Tensor,        # (D, H, W) predicted mask
    omnirad: OmniRadEncoder,
    cfg:     HectorSurvivalConfig,
) -> torch.Tensor:
    """
    OmniRad encoding only — frozen, returns (D, 768) on CPU detached.
    Pooling is NOT done here so GatedAttentionPooling runs inside the
    training loop where the computation graph is live.
    """
    return omnirad.encode_volume_ct_pt(
        ct          = ct,
        pt          = pt,
        mask        = mask,
        num_classes = cfg.num_classes,
        batch_size  = cfg.omni_batch,
    )


def pool_slice_embeddings_hector(
    slice_embs: torch.Tensor,           # (D, 768) on CPU
    pooling:    torch.nn.Module | None,
    cfg:        HectorSurvivalConfig,
    device:     torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pool (D, 768) → (768,) on GPU, keeping grad alive.
    Called inside training loop — no no_grad decorator.
    """
    embs_gpu = slice_embs.to(device)
    if cfg.slice_pooling == "attention" and pooling is not None:
        emb, attn_w = pooling(embs_gpu)
        return emb, attn_w.detach().cpu()
    else:
        return embs_gpu.mean(dim=0), None


@torch.no_grad()
def predict_mask_hector(
    ct:     torch.Tensor,
    pt:     torch.Tensor,
    unet:   torch.nn.Module,
    cfg:    HectorSurvivalConfig,
    device: torch.device,
) -> torch.Tensor:
    ct_pt_vol = torch.stack([ct, pt], dim=1)   # (D, 2, H, W)
    return sliding_window_predict(
        model       = unet,
        volume      = ct_pt_vol,
        num_classes = cfg.num_classes,
        window      = cfg.sw_window,
        stride      = cfg.sw_stride,
        device      = device,
    )


# ─── Main training function ──────────────────────────────────────────────────

def train_survival_hector(cfg: HectorSurvivalConfig | None = None) -> str:
    cfg    = cfg or HectorSurvivalConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    logger_run, csv_path = setup_logging(cfg.log_dir, prefix="survival_hector")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = (
            f"{cfg.experiment_name}_{cfg._variant}_seed{cfg.seed}_fold{cfg.fold_idx}"
            if cfg.fold_idx >= 0 else
            f"{cfg.experiment_name}_{cfg._variant}_seed{cfg.seed}"
        ),
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        notes   = cfg.wandb_notes or None,
        tags    = cfg.wandb_tags  or [],
        reinit  = True,
    )

    logger_run.info("=" * 60)
    logger_run.info("HECKTOR Survival Analysis")
    logger_run.info(f"W&B run  : {run.url}")
    logger_run.info(f"Device   : {device}")
    logger_run.info("=" * 60)
    log_config(logger_run, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _train_ids = getattr(cfg, "_kfold_train_ids", None)
    _val_ids   = getattr(cfg, "_kfold_val_ids",   None)

    _ds_kwargs = dict(
        task2_dir      = cfg.task2_dir,
        metadata_csv   = cfg.metadata_csv,
        split_file     = cfg.split_file,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
    )
    train_ds = HectorTask2Dataset(**_ds_kwargs, mode="train", case_ids=_train_ids)
    val_ds   = HectorTask2Dataset(**_ds_kwargs, mode="val",   case_ids=_val_ids)

    _dl_kwargs = dict(
        batch_size         = 1,
        num_workers        = cfg.num_workers,
        pin_memory         = False,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_dl_kwargs)

    logger_run.info(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
        f"Crop: {cfg.crop_depth} slices (tumour-p={cfg.tumour_crop_p}) | "
        f"Effective batch: {cfg.accumulation_steps}"
    )

    # ── Frozen UNet ────────────────────────────────────────────────────────
    unet = None
    omnirad = None
    pooling = None

    if cfg.use_imaging:
        if not os.path.exists(cfg.unet_ckpt):
            raise FileNotFoundError(
                f"UNet checkpoint not found: '{cfg.unet_ckpt}'\n"
                "Run train_unet_hector.py first."
            )
        unet = SimpleUNet3D(
            n_classes     = cfg.num_classes,
            base_channels = cfg.unet_base_channels,
            trilinear     = cfg.unet_trilinear,
            in_channels   = cfg.in_channels,
        ).to(device)
        ckpt_data = torch.load(cfg.unet_ckpt, map_location=device)
        unet.load_state_dict({
            k.removeprefix("_orig_mod."): v
            for k, v in ckpt_data["model_state"].items()
        })
        unet.eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        logger_run.info(
            f"Frozen UNet loaded  "
            f"(val_mean_dice={ckpt_data.get('val_mean_dice', float('nan')):.4f})"
        )

        omnirad = OmniRadEncoder(device=device, frozen=True)
        logger_run.info("OmniRad loaded and frozen.")

        if cfg.slice_pooling == "attention":
            pooling = GatedAttentionPooling(
                embed_dim   = cfg.embed_dim,
                hidden_size = cfg.attn_hidden_size,
                dropout     = cfg.attn_dropout,
            ).to(device)
            logger_run.info(
                f"GatedAttentionPooling  params: "
                f"{sum(p.numel() for p in pooling.parameters()):,}"
            )
        else:
            logger_run.info("Slice pooling: unweighted mean.")

    # ── Clinical MLP ──────────────────────────────────────────────────────
    clinical_feats_train: dict = {}
    clinical_feats_val:   dict = {}
    clinical_mlp = None

    if cfg.use_clinical:
        cp = ClinicalPreprocessorHector(
            metadata_path     = cfg.metadata_csv,
            missing_threshold = cfg.missing_threshold,
        )
        clinical_feats_train, clinical_feats_val, clin_dim = cp.fit_transform(
            train_ids = train_ds.cases,
            val_ids   = val_ds.cases,
        )
        cp.save(cfg.clinical_preprocessor_path)

        # For HECKTOR use smaller hidden dims given tiny feature space
        hidden_dims = cfg.clinical_hidden_dims or [64, 32]
        clinical_mlp = ClinicalMLP(
            input_dim   = clin_dim,
            output_dim  = cfg.clinical_dim,
            hidden_dims = hidden_dims,
            dropout     = cfg.clinical_dropout,
        ).to(device)
        logger_run.info(
            f"ClinicalMLP  input={clin_dim}  hidden={hidden_dims}  "
            f"output={cfg.clinical_dim}  "
            f"params={sum(p.numel() for p in clinical_mlp.parameters()):,}"
        )
        wandb.log({"model/clinical_mlp_params": sum(p.numel() for p in clinical_mlp.parameters())}, step=0)

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm = EGMDMHead(
        input_size  = cfg.egmdm_input_dim,
        hidden_size = cfg.egmdm_hidden_size,
        E           = cfg.egmdm_E,
        K           = cfg.egmdm_K,
        dropout     = cfg.egmdm_dropout,
    ).to(device)
    n_egmdm = sum(p.numel() for p in egmdm.parameters() if p.requires_grad)
    logger_run.info(f"EGMDMHead  input={cfg.egmdm_input_dim}  params={n_egmdm:,}")
    wandb.log({"model/egmdm_params": n_egmdm}, step=0)

    # ── Optimiser ──────────────────────────────────────────────────────────
    trainable_params = list(egmdm.parameters())
    if pooling is not None:
        trainable_params += list(pooling.parameters())
    if clinical_mlp is not None:
        trainable_params += list(clinical_mlp.parameters())

    criterion = EGMDMLoss(
        lambda_div = cfg.lambda_div,
        lambda_ent = cfg.lambda_ent,
        lambda_mix = cfg.lambda_mix,
    )
    optimizer = optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_cindex    = -1.0
    no_improve_ctr = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        saved = torch.load(cfg.resume_path, map_location=device)
        egmdm.load_state_dict(saved["egmdm_state"])
        if pooling is not None and "pooling_state" in saved:
            pooling.load_state_dict(saved["pooling_state"])
        if clinical_mlp is not None and "clinical_mlp_state" in saved:
            clinical_mlp.load_state_dict(saved["clinical_mlp_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch    = saved["epoch"] + 1
        best_cindex    = saved.get("val_cindex", -1.0)
        no_improve_ctr = saved.get("no_improve_ctr", 0)
        logger_run.info(f"Resumed from epoch {saved['epoch']}")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "train_loss", "train_nll",
            "val_loss", "val_nll", "val_cindex",
        ])

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger_run.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        egmdm.train()
        if pooling is not None:
            pooling.train()
        if clinical_mlp is not None:
            clinical_mlp.train()

        tr_loss = tr_nll = tr_reg_div = tr_reg_ent = 0.0
        tr_img_norm = tr_clin_norm = 0.0
        tr_grad_egmdm = tr_grad_pool = tr_grad_clin = 0.0
        tr_n = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for step, batch in enumerate(pbar, 1):
            ct    = batch["ct"][0]
            pt    = batch["pt"][0]
            event = batch["event"][0]
            t_day = batch["survival_time"][0]

            parts = []

            if cfg.use_imaging:
                # Full volume → UNet mask → crop → OmniRad (no_grad)
                with torch.no_grad():
                    mask = predict_mask_hector(ct, pt, unet, cfg, device)
                    z0, z1   = _tumour_crop(mask, cfg.crop_depth, cfg.tumour_crop_p)
                    ct_enc   = ct[z0:z1]
                    pt_enc   = pt[z0:z1]
                    mask_enc = mask[z0:z1]
                    slice_embs = get_slice_embeddings_hector(
                        ct_enc, pt_enc, mask_enc, omnirad, cfg
                    )

                # Pooling runs with grad in training loop
                img_emb, _ = pool_slice_embeddings_hector(slice_embs, pooling, cfg, device)
                img_emb_n  = F.normalize(img_emb, dim=0)
                parts.append(img_emb_n)
                tr_img_norm += img_emb.norm().item()

            if clinical_mlp is not None:
                cid      = batch["caseid"][0]
                clin_vec = clinical_feats_train.get(cid)
                if clin_vec is not None:
                    clin_emb   = clinical_mlp(clin_vec.to(device))
                    clin_emb_n = F.normalize(clin_emb, dim=0)
                    parts.append(clin_emb_n)
                    tr_clin_norm += clin_emb.norm().item()

            emb = torch.cat(parts, dim=0).unsqueeze(0)  # (1, egmdm_input_dim) on GPU

            params, reg = egmdm(emb)
            t = (t_day / 365.25).unsqueeze(0).to(device)
            e = event.float().unsqueeze(0).to(device)

            loss, nll = criterion(egmdm, params, reg, t, e)
            (loss / cfg.accumulation_steps).backward()

            tr_loss    += loss.item()
            tr_nll     += nll.item()
            tr_reg_div += reg.get("L_div", torch.tensor(0.0)).item()
            tr_reg_ent += reg.get("L_ent", torch.tensor(0.0)).item()

            def _grad_norm(ps):
                g = [p.grad.detach().norm() for p in ps if p.grad is not None]
                return torch.stack(g).norm().item() if g else 0.0

            tr_grad_egmdm += _grad_norm(egmdm.parameters())
            if pooling is not None:
                tr_grad_pool  += _grad_norm(pooling.parameters())
            if clinical_mlp is not None:
                tr_grad_clin  += _grad_norm(clinical_mlp.parameters())
            tr_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Flush remaining
        if len(train_loader) % cfg.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        n = max(tr_n, 1)
        avg_tr_loss    = tr_loss    / len(train_loader)
        avg_tr_nll     = tr_nll     / len(train_loader)
        avg_tr_reg_div = tr_reg_div / len(train_loader)
        avg_tr_reg_ent = tr_reg_ent / len(train_loader)
        avg_img_norm   = tr_img_norm   / n
        avg_clin_norm  = tr_clin_norm  / n
        avg_grad_egmdm = tr_grad_egmdm / n
        avg_grad_pool  = tr_grad_pool  / n
        avg_grad_clin  = tr_grad_clin  / n

        # ── Val — full volume, no crop ─────────────────────────────────────
        egmdm.eval()
        if pooling is not None:
            pooling.eval()
        if clinical_mlp is not None:
            clinical_mlp.eval()

        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []
        all_attn:       list[torch.Tensor] = []
        all_img_norms:  list[float]        = []
        all_clin_norms: list[float]        = []
        all_sigma_mean: list[float]        = []
        all_sigma_min:  list[float]        = []
        all_mix_ent:    list[float]        = []
        all_attn_max:   list[float]        = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct    = batch["ct"][0]
                pt    = batch["pt"][0]
                event = batch["event"][0]
                t_day = batch["survival_time"][0]

                parts   = []
                attn_w  = None
                _img_n = _clin_n = 0.0

                if cfg.use_imaging:
                    mask       = predict_mask_hector(ct, pt, unet, cfg, device)
                    slice_embs = get_slice_embeddings_hector(
                        ct, pt, mask, omnirad, cfg
                    )
                    img_emb, attn_w = pool_slice_embeddings_hector(
                        slice_embs, pooling, cfg, device
                    )
                    _img_n    = img_emb.norm().item()
                    img_emb_n = F.normalize(img_emb, dim=0)
                    parts.append(img_emb_n)

                if clinical_mlp is not None:
                    cid      = batch["caseid"][0]
                    clin_vec = clinical_feats_val.get(cid)
                    if clin_vec is not None:
                        clin_emb   = clinical_mlp(clin_vec.to(device))
                        _clin_n    = clin_emb.norm().item()
                        clin_emb_n = F.normalize(clin_emb, dim=0)
                        parts.append(clin_emb_n)

                all_img_norms.append(_img_n)
                all_clin_norms.append(_clin_n)

                emb = torch.cat(parts, dim=0).unsqueeze(0)

                params, reg = egmdm(emb)
                t = (t_day / 365.25).unsqueeze(0).to(device)
                e = event.float().unsqueeze(0).to(device)

                loss, nll = criterion(egmdm, params, reg, t, e)
                vl_loss += loss.item()
                vl_nll  += nll.item()

                risk = egmdm.cdf(params, torch.tensor([1.0], device=device)).squeeze().cpu()
                all_risks.append(risk)
                all_times.append(t_day.cpu())
                all_events.append(event.cpu())
                if attn_w is not None:
                    all_attn.append(attn_w)
                    all_attn_max.append(attn_w.max().item())

                all_sigma_mean.append(params["sigma"].mean().item())
                all_sigma_min.append(params["sigma"].min().item())
                w = params["w"].clamp(1e-8)
                all_mix_ent.append(-(w * w.log()).sum(-1).mean().item())

        avg_vl_loss = vl_loss / len(val_loader)
        avg_vl_nll  = vl_nll  / len(val_loader)
        c_index = concordance_index(
            torch.stack(all_risks),
            torch.stack(all_times),
            torch.stack(all_events),
        )
        scheduler.step()

        logger_run.info(f"  Train → Loss: {avg_tr_loss:.4f}  NLL: {avg_tr_nll:.4f}")
        logger_run.info(
            f"  Val   → Loss: {avg_vl_loss:.4f}  NLL: {avg_vl_nll:.4f}  "
            f"C-index: {c_index:.4f}  (best: {best_cindex:.4f})"
        )

        def _mean(lst): return sum(lst) / max(len(lst), 1)

        wb_log = {
            "train/lr":              lr,
            "train/loss":            avg_tr_loss,
            "train/nll":             avg_tr_nll,
            "train/reg_div":         avg_tr_reg_div,
            "train/reg_ent":         avg_tr_reg_ent,
            "train/img_emb_norm":    avg_img_norm,
            "train/clin_emb_norm":   avg_clin_norm,
            "train/grad_egmdm":      avg_grad_egmdm,
            "train/grad_pooling":    avg_grad_pool,
            "train/grad_clinical":   avg_grad_clin,
            "val/loss":              avg_vl_loss,
            "val/nll":               avg_vl_nll,
            "val/cindex":            c_index,
            "val/best_cindex":       max(best_cindex, c_index),
            "val/sigma_mean":        _mean(all_sigma_mean),
            "val/sigma_min":         _mean(all_sigma_min),
            "val/mixture_entropy":   _mean(all_mix_ent),
            "val/attn_max_weight":   _mean(all_attn_max) if all_attn_max else 0.0,
            "val/img_emb_norm":      _mean(all_img_norms),
            "val/clin_emb_norm":     _mean(all_clin_norms),
        }
        if all_attn:
            wb_log["val/attn_entropy"] = _mean([
                -(w * (w + 1e-8).log()).sum().item() for w in all_attn
            ])

        wandb.log(wb_log, step=epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{avg_tr_loss:.6f}", f"{avg_tr_nll:.6f}",
                f"{avg_vl_loss:.6f}", f"{avg_vl_nll:.6f}", f"{c_index:.4f}",
            ])

        ckpt_payload = {
            "epoch":           epoch,
            "egmdm_state":     egmdm.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        avg_vl_loss,
            "val_cindex":      c_index,
            "no_improve_ctr":  no_improve_ctr,
            "config":          cfg.to_dict(),
        }
        if pooling is not None:
            ckpt_payload["pooling_state"] = pooling.state_dict()
        if clinical_mlp is not None:
            ckpt_payload["clinical_mlp_state"] = clinical_mlp.state_dict()

        torch.save(ckpt_payload, cfg.last_ckpt)

        if c_index > best_cindex:
            best_cindex    = c_index
            no_improve_ctr = 0
            torch.save(ckpt_payload, cfg.best_ckpt)
            logger_run.info(f"  ✓ Best saved → {cfg.best_ckpt}  (C-index={best_cindex:.4f})")
            wandb.run.summary.update({
                "best_val_cindex": best_cindex,
                "best_val_loss":   avg_vl_loss,
                "best_epoch":      epoch,
            })
        else:
            no_improve_ctr += 1
            logger_run.info(
                f"  No improvement {no_improve_ctr}/{cfg.early_stop_patience}"
            )

        if no_improve_ctr >= cfg.early_stop_patience:
            logger_run.info(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["early_stop_epoch"] = epoch
            break

    logger_run.info(f"\nDone. Best val C-index: {best_cindex:.4f}")
    wandb.finish()
    return cfg.best_ckpt


# ─── K-fold wrapper ───────────────────────────────────────────────────────────

def train_survival_hector_kfold(cfg: HectorSurvivalConfig | None = None) -> dict:
    """
    K-fold CV for HECKTOR survival.

    Stratification by Relapse status.  Events loaded from metadata CSV.
    All case IDs come from both task2 train+val in the split file.
    """
    import json

    cfg = cfg or HectorSurvivalConfig()
    assert cfg.use_kfold

    log = logging.getLogger(__name__)

    # Collect all Task 2 case IDs
    with open(cfg.split_file) as f:
        splits = json.load(f)
    task2 = splits.get("task2", splits)
    all_ids = task2.get("train", []) + task2.get("val", [])

    # Event labels from CSV
    import pandas as pd, numpy as np
    df = pd.read_csv(cfg.metadata_csv).set_index("PatientID")
    events = np.array([int(df.loc[cid, "Relapse"]) for cid in all_ids])
    log.info(f"HECKTOR Task 2: {len(all_ids)} patients, event rate={events.mean():.1%}")

    folds = make_kfold_splits(all_ids, events, n_folds=cfg.n_folds, seed=cfg.seed)

    fold_cindices: list[float] = []
    best_ckpts:    list[str]   = []

    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        log.info(
            f"\n{'#'*65}\n  Fold {fold_idx+1}/{cfg.n_folds}  "
            f"train={len(train_ids)}  val={len(val_ids)}\n{'#'*65}"
        )
        fold_cfg = dataclasses.replace(cfg, fold_idx=fold_idx)
        fold_cfg._kfold_train_ids = train_ids
        fold_cfg._kfold_val_ids   = val_ids

        best_ckpt = train_survival_hector(fold_cfg)
        best_ckpts.append(best_ckpt)

        ckpt        = torch.load(best_ckpt, map_location="cpu")
        best_cindex = ckpt.get("val_cindex", float("nan"))
        fold_cindices.append(best_cindex)
        log.info(f"  Fold {fold_idx+1} → C-index: {best_cindex:.4f}")

    arr  = np.array(fold_cindices)
    mean = float(arr.mean())
    std  = float(arr.std())

    log.info(f"\n{'='*65}")
    log.info(f"  HECKTOR {cfg.n_folds}-FOLD  (seed={cfg.seed})")
    log.info(f"{'='*65}")
    for i, ci in enumerate(fold_cindices):
        log.info(f"  Fold {i:2d}  C-index: {ci:.4f}")
    log.info(f"  Mean: {mean:.4f} ± {std:.4f}")
    log.info(f"{'='*65}\n")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    summary_run = wandb.init(
        project = cfg.wandb_project,
        name    = (
            f"{cfg.experiment_name}_{cfg._variant}_seed{cfg.seed}"
            f"_{cfg.n_folds}fold_summary"
        ),
        group   = cfg.experiment_name,
        config  = {**cfg.to_dict(), "fold_cindices": str(fold_cindices)},
        notes   = cfg.wandb_notes or None,
        tags    = (cfg.wandb_tags or []) + ["kfold_summary"],
        reinit  = True,
    )
    for i, ci in enumerate(fold_cindices):
        wandb.log({"fold/cindex": ci, "fold/index": i}, step=i)
    wandb.log({"kfold/mean_cindex": mean, "kfold/std_cindex": std})
    summary_run.summary.update({
        "kfold_mean_cindex":  mean,
        "kfold_std_cindex":   std,
        "best_fold":          int(arr.argmax()),
        "best_fold_cindex":   float(arr.max()),
        "worst_fold_cindex":  float(arr.min()),
    })
    wandb.finish()

    return {"fold_cindices": fold_cindices, "mean_cindex": mean,
            "std_cindex": std, "best_ckpts": best_ckpts}


if __name__ == "__main__":
    _cfg = HectorSurvivalConfig()
    if "--kfold" in sys.argv:
        _cfg.use_kfold = True
        train_survival_hector_kfold(_cfg)
    else:
        train_survival_hector(_cfg)