"""
train_survival_rgb.py — RGB-OmniRad survival analysis
======================================================
Run single experiment:
    python train_survival_rgb.py

Pipeline
--------
  KitsDatasetRGB  →  (D, 3, H, W) 3-channel HU-windowed CT  (no mask, no UNet)
        ↓
  OmniRad.encode_volume_rgb()  [frozen]    →  (D, embed_dim)
        ↓
  GatedAttentionPooling  [trainable]       →  (embed_dim,)
    or unweighted mean
        ↓
  EGMDMHead              [trainable]       →  survival distribution

Key differences from train_survival.py
---------------------------------------
  - No UNet, no segmentation mask, no sliding-window inference.
  - OmniRad receives 3-channel HU-windowed CT via encode_volume_rgb().
  - Dataset is KitsDatasetRGB — returns full volumes in both train and val,
    no tumour-centred cropping (gated attention handles variable depth).
  - Config is SurvivalRGBConfig.

Early stopping: val C-index (higher = better).
"""

import csv
import os
os.environ["WANDB_MODE"] = "offline"

import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.survival_rgb_config import SurvivalRGBConfig
from data.dataset                import KitsDatasetRGB
from losses.survival_loss        import EGMDMLoss
from models.egmdm                import EGMDMHead
from models.omnirad              import GatedAttentionPooling, OmniRadEncoder
from utils.logging_utils         import log_config, setup_logging
from utils.metrics               import concordance_index
from utils.seed                  import set_seed


# ─── Patient embedding pipeline ───────────────────────────────────────────────

@torch.no_grad()
def embed_patient_rgb(
    ct_rgb:  torch.Tensor,                        # (D, 3, H, W) float32 on CPU
    omnirad: OmniRadEncoder,
    pooling: torch.nn.Module | None,
    cfg:     SurvivalRGBConfig,
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build a patient-level embedding from a 3-channel HU-windowed CT volume.

    Pipeline
    --------
    1. OmniRad encodes every slice via encode_volume_rgb() → (D, embed_dim) on CPU.
    2. Pool across depth:
         "mean"      → unweighted mean → (embed_dim,) on CPU
         "attention" → gated attention → (embed_dim,) on CPU
                       also returns (D,) attention weights for logging.

    Parameters
    ----------
    ct_rgb  : (D, 3, H, W) float32, channels already in [0, 1], on CPU
    omnirad : frozen OmniRadEncoder
    pooling : GatedAttentionPooling (trainable) or None (mean pooling)
    cfg     : SurvivalRGBConfig
    device  : GPU device for attention forward pass

    Returns
    -------
    embedding    : (embed_dim,) on CPU
    attn_weights : (D,) on CPU if slice_pooling="attention", else None
    """
    # Per-slice embeddings — no mask assembly, RGB channels go in directly
    slice_embs = omnirad.encode_volume_rgb(
        ct_rgb     = ct_rgb,
        batch_size = cfg.omni_batch,
    )   # (D, embed_dim) on CPU

    if cfg.slice_pooling == "attention":
        # Attention is trainable — re-enable grad inside the no_grad scope
        with torch.enable_grad():
            emb, attn_weights = pooling(slice_embs.to(device))
        return emb.cpu(), attn_weights.detach().cpu()
    else:
        return slice_embs.mean(dim=0), None   # (embed_dim,) on CPU


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_survival_rgb(cfg: SurvivalRGBConfig | None = None) -> str:
    """
    Train one RGB-OmniRad survival experiment defined by `cfg`.

    Returns
    -------
    best_ckpt_path : str
    """
    cfg    = cfg or SurvivalRGBConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)

    logger, csv_path = setup_logging(cfg.log_dir, prefix="survival_rgb")

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_seed{cfg.seed}",
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        reinit  = True,
    )

    logger.info("=" * 60)
    logger.info(f"RGB-OmniRad Survival Analysis")
    logger.info(f"W&B run  : {run.url}")
    logger.info(f"Device   : {device}")
    logger.info("=" * 60)
    log_config(logger, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        rootdir        = cfg.root_dir,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        split_file     = cfg.json_path,
        metadata_path  = os.path.join(cfg.root_dir, "kits.json"),
    )
    # KitsDatasetRGB always returns full volumes (no cropping) — mode only
    # controls which case IDs are loaded (train split vs val split).
    train_ds = KitsDatasetRGB(**_ds_kwargs, mode="train")
    val_ds   = KitsDatasetRGB(**_ds_kwargs, mode="val")

    _dl_kwargs = dict(
        batch_size         = 1,
        num_workers        = cfg.num_workers,
        pin_memory         = False,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_dl_kwargs)

    logger.info(f"Train: {len(train_ds)} cases | Val: {len(val_ds)} cases")
    logger.info(
        f"HU windows — "
        f"R: {KitsDatasetRGB.HU_WINDOWS[0]}  "
        f"G: {KitsDatasetRGB.HU_WINDOWS[1]}  "
        f"B: {KitsDatasetRGB.HU_WINDOWS[2]}"
    )

    # ── Frozen OmniRad ─────────────────────────────────────────────────────
    omnirad = OmniRadEncoder(device=device, frozen=True)
    logger.info("OmniRad loaded and frozen.")

    # ── Slice pooling (trainable when attention) ───────────────────────────
    if cfg.slice_pooling == "attention":
        pooling = GatedAttentionPooling(
            embed_dim   = cfg.embed_dim,
            hidden_size = cfg.attn_hidden_size,
            dropout     = cfg.attn_dropout,
        ).to(device)
        n_attn = sum(p.numel() for p in pooling.parameters())
        logger.info(f"GatedAttentionPooling  params: {n_attn:,}")
    elif cfg.slice_pooling == "mean":
        pooling = None
        logger.info("Slice pooling: unweighted mean (no extra parameters).")
    else:
        raise ValueError(
            f"Unknown slice_pooling='{cfg.slice_pooling}'. "
            "Choose 'mean' or 'attention'."
        )

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm = EGMDMHead(
        input_size  = cfg.embed_dim,
        hidden_size = cfg.egmdm_hidden_size,
        E           = cfg.egmdm_E,
        K           = cfg.egmdm_K,
        dropout     = cfg.egmdm_dropout,
    ).to(device)

    n_egmdm = sum(p.numel() for p in egmdm.parameters() if p.requires_grad)
    logger.info(f"EGMDMHead              params: {n_egmdm:,}")
    wandb.log({"model/egmdm_params": n_egmdm}, step=0)

    # ── Loss / optimiser / scheduler ───────────────────────────────────────
    # Only pooling + EGMDM are trained; OmniRad is fully frozen.
    trainable_params = list(egmdm.parameters())
    if pooling is not None:
        trainable_params += list(pooling.parameters())

    criterion = EGMDMLoss(lambda_div=cfg.lambda_div, lambda_ent=cfg.lambda_ent)
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
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch    = saved["epoch"] + 1
        best_cindex    = saved.get("val_cindex", -1.0)
        no_improve_ctr = saved.get("no_improve_ctr", 0)
        logger.info(f"Resumed from '{cfg.resume_path}' (epoch {saved['epoch']})")

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

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for batch in pbar:
            ct_rgb  = batch["ct_rgb"][0]          # (D, 3, H, W) CPU
            event   = batch["event"][0]
            t_day   = batch["survival_time"][0]

            emb, _ = embed_patient_rgb(
                ct_rgb  = ct_rgb,
                omnirad = omnirad,
                pooling = pooling,
                cfg     = cfg,
                device  = device,
            )
            emb = emb.unsqueeze(0).to(device)     # (1, embed_dim)

            params, reg = egmdm(emb)
            t = (t_day / 365.25).unsqueeze(0).to(device)
            e = event.float().unsqueeze(0).to(device)

            loss, nll = criterion(egmdm, params, reg, t, e)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_nll  += nll.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", nll=f"{nll.item():.4f}")

        avg_tr_loss = tr_loss / len(train_loader)
        avg_tr_nll  = tr_nll  / len(train_loader)

        # ── Val ────────────────────────────────────────────────────────────
        egmdm.eval()
        if pooling is not None:
            pooling.eval()
        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []
        all_attn: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct_rgb  = batch["ct_rgb"][0]
                event   = batch["event"][0]
                t_day   = batch["survival_time"][0]

                emb, attn_w = embed_patient_rgb(
                    ct_rgb  = ct_rgb,
                    omnirad = omnirad,
                    pooling = pooling,
                    cfg     = cfg,
                    device  = device,
                )
                emb = emb.unsqueeze(0).to(device)

                params, reg = egmdm(emb)
                t = (t_day / 365.25).unsqueeze(0).to(device)
                e = event.float().unsqueeze(0).to(device)

                loss, nll = criterion(egmdm, params, reg, t, e)
                vl_loss += loss.item()
                vl_nll  += nll.item()

                # Risk = P(event by 1 year)
                risk = egmdm.cdf(params, torch.tensor([1.0], device=device)).squeeze().cpu()
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

        # ── Logging ────────────────────────────────────────────────────────
        logger.info(f"  Train → Loss: {avg_tr_loss:.4f}  NLL: {avg_tr_nll:.4f}")
        logger.info(
            f"  Val   → Loss: {avg_vl_loss:.4f}  NLL: {avg_vl_nll:.4f}  "
            f"C-index: {c_index:.4f}  (best: {best_cindex:.4f})"
        )

        wb_log: dict = {
            "train/lr":        lr,
            "train/loss":      avg_tr_loss,
            "train/nll":       avg_tr_nll,
            "val/loss":        avg_vl_loss,
            "val/nll":         avg_vl_nll,
            "val/cindex":      c_index,
            "val/best_cindex": max(best_cindex, c_index),
        }
        # Attention entropy — per-patient then averaged (variable D per patient)
        if all_attn:
            entropy = torch.tensor([
                -(w * (w + 1e-8).log()).sum().item()
                for w in all_attn
            ]).mean()
            wb_log["val/attn_entropy"] = entropy.item()

        wandb.log(wb_log, step=epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{avg_tr_loss:.6f}", f"{avg_tr_nll:.6f}",
                f"{avg_vl_loss:.6f}", f"{avg_vl_nll:.6f}", f"{c_index:.4f}",
            ])

        # ── Checkpoint ─────────────────────────────────────────────────────
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
                f"  ✓ Best model saved → {cfg.best_ckpt}  "
                f"(C-index={best_cindex:.4f}  val_loss={avg_vl_loss:.4f})"
            )
            wandb.run.summary.update({
                "best_val_cindex": best_cindex,
                "best_val_loss":   avg_vl_loss,
                "best_epoch":      epoch,
            })
        else:
            no_improve_ctr += 1
            logger.info(
                f"  No improvement {no_improve_ctr}/{cfg.early_stop_patience}  "
                f"(best C-index={best_cindex:.4f})"
            )

        if no_improve_ctr >= cfg.early_stop_patience:
            logger.info(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["early_stop_epoch"] = epoch
            break

    logger.info(f"\nRGB experiment complete. Best val C-index: {best_cindex:.4f}")
    logger.info(f"Best checkpoint: {cfg.best_ckpt}")
    wandb.finish()

    return cfg.best_ckpt


if __name__ == "__main__":
    train_survival_rgb()