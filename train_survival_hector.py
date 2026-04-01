"""
train_survival_hector.py — HECKTOR Task 2 survival training
============================================================
Run:
    python train_survival_hector.py

Pipeline
--------
  HectorTask2Dataset (CT + PT, no mask)
    → 2-channel UNet predicts tumour mask  [frozen, trained on Task 1]

  Training (cropped for speed):
    → crop cfg.crop_depth slices centred on tumour (prob p) or randomly
    → OmniRad.encode_volume_ct_pt() on the crop   [frozen] → (16, 768)
    → GatedAttentionPooling                        [trainable] → (768,)
    → EGMDMHead                                    [trainable] → RFS

  Validation (full volume):
    → OmniRad encodes all D slices                 → (D, 768)
    → GatedAttentionPooling                        → (768,)
    → EGMDMHead

Why cropping only during training?
-----------------------------------
The UNet runs on the full volume regardless (to locate tumour).  After that,
only cfg.crop_depth slices are passed to OmniRad during training — reducing
ViT forward passes from ~150 to 16 per patient.  Gradient accumulation over
cfg.accumulation_steps patients simulates a larger effective batch size.
Validation uses the full volume to measure real-world performance.

Early stopping: val C-index (higher = better).
"""

import csv
import os
import random
os.environ["WANDB_MODE"] = "offline"
import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.hector_survival_config import HectorSurvivalConfig
from data.dataset_hector            import HectorTask2Dataset
from losses.survival_loss           import EGMDMLoss
from models.egmdm                   import EGMDMHead
from models.omnirad                 import GatedAttentionPooling, OmniRadEncoder
from models.unet                    import SimpleUNet3D
from utils.inference                import sliding_window_predict
from utils.logging_utils            import log_config, setup_logging
from utils.metrics                  import concordance_index
from utils.seed                     import set_seed


# ─── Crop helper ─────────────────────────────────────────────────────────────

def _tumour_crop(
    mask:       torch.Tensor,   # (D, H, W) int64 — predicted mask
    crop_depth: int,
    p:          float,
) -> tuple[int, int]:
    """
    Return (z_start, z_end) for a depth crop of `crop_depth` slices.

    With probability p: centre the crop on a tumour slice (label > 0).
    Otherwise          : pick a random start position.
    """
    D = mask.shape[0]

    tumour_slices = (mask > 0).any(dim=(1, 2))
    indices       = torch.where(tumour_slices)[0]

    if len(indices) > 0 and torch.rand(()).item() < p:
        center = indices[random.randint(0, len(indices) - 1)].item()
        z_min  = max(0, center - crop_depth + 1)
        z_max  = min(center, D - crop_depth)
        z      = (
            max(0, min(center, D - crop_depth))
            if z_max < z_min
            else random.randint(z_min, z_max)
        )
    else:
        z = random.randint(0, max(0, D - crop_depth))

    return z, z + crop_depth


# ─── Patient embedding ────────────────────────────────────────────────────────

@torch.no_grad()
def embed_patient_hector(
    ct:         torch.Tensor,           # (D, H, W) float32 [0,1] on CPU
    pt:         torch.Tensor,           # (D, H, W) float32 [0,1] on CPU
    unet:       torch.nn.Module,
    omnirad:    OmniRadEncoder,
    pooling:    torch.nn.Module | None,
    cfg:        HectorSurvivalConfig,
    device:     torch.device,
    is_training:bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build a patient-level embedding from CT + PT volumes.

    Training  : UNet on full volume → tumour-aware 16-slice crop → OmniRad
    Validation: UNet on full volume → OmniRad on all slices

    Returns
    -------
    embedding    : (embed_dim,) on CPU
    attn_weights : (D_crop,) or (D,) on CPU if attention pooling, else None
    """
    # ── Step 1: full-volume UNet mask prediction ───────────────────────────
    ct_pt_vol = torch.stack([ct, pt], dim=1)   # (D, 2, H, W)
    mask = sliding_window_predict(
        model       = unet,
        volume      = ct_pt_vol,
        num_classes = cfg.num_classes,
        window      = cfg.sw_window,
        stride      = cfg.sw_stride,
        device      = device,
    )   # (D, H, W) int64 on CPU

    # ── Step 2: optionally crop to fixed depth for training ───────────────
    if is_training:
        z0, z1 = _tumour_crop(mask, cfg.crop_depth, cfg.tumour_crop_p)
        ct_enc   = ct[z0:z1]      # (crop_depth, H, W)
        pt_enc   = pt[z0:z1]
        mask_enc = mask[z0:z1]
    else:
        ct_enc   = ct             # (D, H, W) — full volume for val
        pt_enc   = pt
        mask_enc = mask

    # ── Step 3: OmniRad per-slice embeddings ─────────────────────────────
    slice_embs = omnirad.encode_volume_ct_pt(
        ct          = ct_enc,
        pt          = pt_enc,
        mask        = mask_enc,
        num_classes = cfg.num_classes,
        batch_size  = cfg.omni_batch,
    )   # (crop_depth|D, 768) on CPU

    # ── Step 4: pool across depth ─────────────────────────────────────────
    if cfg.slice_pooling == "attention":
        with torch.enable_grad():
            emb, attn_weights = pooling(slice_embs.to(device))
        return emb.cpu(), attn_weights.detach().cpu()
    else:
        return slice_embs.mean(dim=0), None


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_survival_hector(cfg: HectorSurvivalConfig | None = None) -> str:
    cfg    = cfg or HectorSurvivalConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    logger, csv_path = setup_logging(cfg.log_dir, prefix="survival_hector")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_seed{cfg.seed}",
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        reinit  = True,
    )

    logger.info("=" * 60)
    logger.info("HECKTOR Survival Analysis")
    logger.info(f"W&B run  : {run.url}")
    logger.info(f"Device   : {device}")
    logger.info("=" * 60)
    log_config(logger, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        task2_dir      = cfg.task2_dir,
        metadata_csv   = cfg.metadata_csv,
        split_file     = cfg.split_file,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
    )
    train_ds = HectorTask2Dataset(**_ds_kwargs, mode="train")
    val_ds   = HectorTask2Dataset(**_ds_kwargs, mode="val")

    # batch_size=1 for both — full volumes have variable D.
    # Effective batch size during training = accumulation_steps.
    _dl_kwargs = dict(
        batch_size         = 1,
        num_workers        = cfg.num_workers,
        pin_memory         = False,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_dl_kwargs)

    logger.info(
        f"Train: {len(train_ds)} cases | Val: {len(val_ds)} cases\n"
        f"Training crop: {cfg.crop_depth} slices  "
        f"(tumour-centred p={cfg.tumour_crop_p})  "
        f"effective batch={cfg.accumulation_steps}"
    )

    # ── Frozen 2-channel UNet ──────────────────────────────────────────────
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
    unet_ckpt = torch.load(cfg.unet_ckpt, map_location=device)
    unet.load_state_dict({
        k.removeprefix("_orig_mod."): v
        for k, v in unet_ckpt["model_state"].items()
    })
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    logger.info(
        f"Frozen UNet loaded  "
        f"(val_mean_dice={unet_ckpt.get('val_mean_dice', float('nan')):.4f})"
    )

    # ── Frozen OmniRad ─────────────────────────────────────────────────────
    omnirad = OmniRadEncoder(device=device, frozen=True)
    logger.info("OmniRad loaded and frozen.")

    # ── Slice pooling ──────────────────────────────────────────────────────
    if cfg.slice_pooling == "attention":
        pooling = GatedAttentionPooling(
            embed_dim   = cfg.embed_dim,
            hidden_size = cfg.attn_hidden_size,
            dropout     = cfg.attn_dropout,
        ).to(device)
        logger.info(
            f"GatedAttentionPooling  params: "
            f"{sum(p.numel() for p in pooling.parameters()):,}"
        )
    elif cfg.slice_pooling == "mean":
        pooling = None
        logger.info("Slice pooling: unweighted mean.")
    else:
        raise ValueError(f"Unknown slice_pooling='{cfg.slice_pooling}'.")

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm = EGMDMHead(
        input_size  = cfg.embed_dim,
        hidden_size = cfg.egmdm_hidden_size,
        E           = cfg.egmdm_E,
        K           = cfg.egmdm_K,
        dropout     = cfg.egmdm_dropout,
    ).to(device)
    logger.info(
        f"EGMDMHead              params: "
        f"{sum(p.numel() for p in egmdm.parameters() if p.requires_grad):,}"
    )

    # ── Optimiser ──────────────────────────────────────────────────────────
    trainable_params = list(egmdm.parameters())
    if pooling is not None:
        trainable_params += list(pooling.parameters())

    criterion = EGMDMLoss(lambda_div=cfg.lambda_div, lambda_ent=cfg.lambda_ent)
    optimizer  = optim.AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_cindex    = -1.0
    no_improve_ctr = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        saved = torch.load(cfg.resume_path, map_location=device)
        egmdm.load_state_dict(saved["egmdm_state"])
        if pooling is not None and "pooling_state" in saved:
            pooling.load_state_dict(saved["pooling_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch    = saved["epoch"] + 1
        best_cindex    = saved.get("val_cindex", -1.0)
        no_improve_ctr = saved.get("no_improve_ctr", 0)
        logger.info(f"Resumed from epoch {saved['epoch']}")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_nll",
            "val_loss", "val_nll", "val_cindex",
        ])

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        egmdm.train()
        if pooling is not None:
            pooling.train()
        tr_loss = tr_nll = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for step, batch in enumerate(pbar, 1):
            ct    = batch["ct"][0]            # (D, H, W) CPU
            pt    = batch["pt"][0]
            event = batch["event"][0]
            t_day = batch["survival_time"][0]

            # UNet → crop 16 slices → OmniRad (16 slices only, not full D)
            emb, _ = embed_patient_hector(
                ct=ct, pt=pt, unet=unet, omnirad=omnirad,
                pooling=pooling, cfg=cfg, device=device,
                is_training=True,
            )
            emb = emb.unsqueeze(0).to(device)   # (1, 768)

            params, reg = egmdm(emb)
            t = (t_day / 365.25).unsqueeze(0).to(device)
            e = event.float().unsqueeze(0).to(device)

            loss, nll = criterion(egmdm, params, reg, t, e)

            # Scale loss for accumulation — gradient accumulates across
            # accumulation_steps patients before each optimiser step.
            (loss / cfg.accumulation_steps).backward()

            tr_loss += loss.item()
            tr_nll  += nll.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining gradients at epoch end
        if len(train_loader) % cfg.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_tr_loss = tr_loss / len(train_loader)
        avg_tr_nll  = tr_nll  / len(train_loader)

        # ── Val — full volume, no cropping ─────────────────────────────────
        egmdm.eval()
        if pooling is not None:
            pooling.eval()
        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []
        all_attn: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct    = batch["ct"][0]
                pt    = batch["pt"][0]
                event = batch["event"][0]
                t_day = batch["survival_time"][0]

                emb, attn_w = embed_patient_hector(
                    ct=ct, pt=pt, unet=unet, omnirad=omnirad,
                    pooling=pooling, cfg=cfg, device=device,
                    is_training=False,   # full volume
                )
                emb = emb.unsqueeze(0).to(device)

                params, reg = egmdm(emb)
                t = (t_day / 365.25).unsqueeze(0).to(device)
                e = event.float().unsqueeze(0).to(device)

                loss, nll = criterion(egmdm, params, reg, t, e)
                vl_loss += loss.item()
                vl_nll  += nll.item()

                risk = egmdm.cdf(
                    params, torch.tensor([1.0], device=device)
                ).squeeze().cpu()
                all_risks.append(risk)
                all_times.append(t_day.cpu())
                all_events.append(event.cpu())
                if attn_w is not None:
                    all_attn.append(attn_w)

        avg_vl_loss = vl_loss / len(val_loader)
        avg_vl_nll  = vl_nll  / len(val_loader)
        c_index = concordance_index(
            torch.stack(all_risks),
            torch.stack(all_times),
            torch.stack(all_events),
        )
        scheduler.step()

        logger.info(f"  Train → Loss: {avg_tr_loss:.4f}  NLL: {avg_tr_nll:.4f}")
        logger.info(
            f"  Val   → Loss: {avg_vl_loss:.4f}  NLL: {avg_vl_nll:.4f}  "
            f"C-index: {c_index:.4f}  (best: {best_cindex:.4f})"
        )

        wb_log = {
            "train/lr":        lr,
            "train/loss":      avg_tr_loss,
            "train/nll":       avg_tr_nll,
            "val/loss":        avg_vl_loss,
            "val/nll":         avg_vl_nll,
            "val/cindex":      c_index,
            "val/best_cindex": max(best_cindex, c_index),
        }
        if all_attn:
            entropy = torch.tensor([
                -(w * (w + 1e-8).log()).sum().item() for w in all_attn
            ]).mean()
            wb_log["val/attn_entropy"] = entropy.item()

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

        torch.save(ckpt_payload, cfg.last_ckpt)

        if c_index > best_cindex:
            best_cindex    = c_index
            no_improve_ctr = 0
            torch.save(ckpt_payload, cfg.best_ckpt)
            logger.info(
                f"  ✓ Best saved → {cfg.best_ckpt}  (C-index={best_cindex:.4f})"
            )
            wandb.run.summary.update({
                "best_val_cindex": best_cindex,
                "best_val_loss":   avg_vl_loss,
                "best_epoch":      epoch,
            })
        else:
            no_improve_ctr += 1
            logger.info(
                f"  No improvement {no_improve_ctr}/{cfg.early_stop_patience}"
            )

        if no_improve_ctr >= cfg.early_stop_patience:
            logger.info(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["early_stop_epoch"] = epoch
            break

    logger.info(f"\nDone. Best val C-index: {best_cindex:.4f}")
    wandb.finish()
    return cfg.best_ckpt


if __name__ == "__main__":
    train_survival_hector()