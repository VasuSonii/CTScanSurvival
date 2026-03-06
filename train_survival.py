"""
train_survival.py — Phase 2: EGMDM survival analysis
======================================================
Run single experiment:
    python -m kits21.train_survival

Run via launcher (multiple seeds / hyperparams):
    python -m kits21.run_experiments

All hyperparameters live in kits21/configs/survival_config.py.
Each run writes exclusively to its own run_dir:
    runs/<experiment_name>/seed_<seed>/
"""

import csv
import os
from datetime import datetime

import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.survival_config import SurvivalConfig
from data.dataset            import KitsDataset
from losses.survival_loss    import EGMDMLoss
from models.egmdm            import EGMDMHead
from models.omnirad          import OmniRadEncoder
from models.unet             import SimpleUNet3D
from utils.inference         import sliding_window_predict
from utils.logging_utils     import setup_logging
from utils.metrics           import concordance_index
from utils.seed              import set_seed


# ─── Patient embedding pipeline ───────────────────────────────────────────────

@torch.no_grad()
def embed_patient(
    ct:      torch.Tensor,
    unet:    torch.nn.Module,
    omnirad: OmniRadEncoder,
    cfg:     SurvivalConfig,
    device:  torch.device,
) -> torch.Tensor:
    """
    CT (D, H, W) → UNet mask → OmniRad per-slice embeddings → mean-pooled (embed_dim,).
    Everything returned on CPU.
    """
    pred_mask = sliding_window_predict(
        model=unet, volume=ct, num_classes=cfg.num_classes,
        window=cfg.sw_window, stride=cfg.sw_stride, device=device,
    )
    slice_embs = omnirad.encode_volume(
        ct=ct, mask=pred_mask, num_classes=cfg.num_classes, batch_size=cfg.omni_batch,
    )
    return slice_embs.mean(dim=0)   # (embed_dim,)


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_survival(cfg: SurvivalConfig | None = None) -> str:
    """
    Train one survival experiment defined by `cfg`.

    Returns
    -------
    best_ckpt_path : str
    """
    cfg    = cfg or SurvivalConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Seed everything ────────────────────────────────────────────────────
    set_seed(cfg.seed)

    logger, csv_path = setup_logging(cfg.log_dir, prefix="survival")

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
    logger.info(f"Phase 2 — EGMDM Survival Analysis")
    logger.info(f"Experiment : {cfg.experiment_name}  |  Seed: {cfg.seed}")
    logger.info(f"Run dir    : {cfg.run_dir}")
    logger.info(f"UNet ckpt  : {cfg.unet_ckpt}")
    logger.info(f"W&B run    : {run.url}")
    logger.info(f"Device     : {device}")
    logger.info("=" * 60)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        rootdir        = cfg.root_dir,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        split_file     = cfg.json_path,
        metadata_path  = os.path.join(cfg.root_dir, "kits.json"),
    )
    train_ds = KitsDataset(**_ds_kwargs, mode="train")
    val_ds   = KitsDataset(**_ds_kwargs, mode="val")

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

    # ── Frozen UNet ────────────────────────────────────────────────────────
    if not os.path.exists(cfg.unet_ckpt):
        raise FileNotFoundError(
            f"UNet checkpoint not found: '{cfg.unet_ckpt}'\n"
            "Run Phase 1 (train_unet) first, or set cfg.unet_ckpt correctly."
        )
    unet = SimpleUNet3D(
        n_classes=cfg.num_classes,
        base_channels=cfg.unet_base_channels,
        trilinear=cfg.unet_trilinear,
    ).to(device)
    saved = torch.load(cfg.unet_ckpt, map_location=device)
    unet.load_state_dict(saved["model_state"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    logger.info(f"Frozen UNet loaded  (trained val_mean_dice={saved.get('val_mean_dice', '?'):.4f})")

    # ── Frozen OmniRad ─────────────────────────────────────────────────────
    omnirad = OmniRadEncoder(device=device, frozen=True)
    logger.info("OmniRad loaded and frozen.")

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm = EGMDMHead(
        input_size  = cfg.embed_dim,
        hidden_size = cfg.egmdm_hidden_size,
        E           = cfg.egmdm_E,
        K           = cfg.egmdm_K,
        dropout     = cfg.egmdm_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in egmdm.parameters() if p.requires_grad)
    logger.info(f"EGMDMHead trainable params: {n_params:,}")
    wandb.log({"model/trainable_params": n_params}, step=0)

    # ── Loss / optimiser / scheduler ───────────────────────────────────────
    criterion = EGMDMLoss(lambda_div=cfg.lambda_div, lambda_ent=cfg.lambda_ent)
    optimizer = optim.AdamW(egmdm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float("inf")
    no_improve_ctr = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        saved = torch.load(cfg.resume_path, map_location=device)
        egmdm.load_state_dict(saved["egmdm_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch    = saved["epoch"] + 1
        best_val_loss  = saved.get("val_loss", float("inf"))
        no_improve_ctr = saved.get("no_improve_ctr", 0)
        logger.info(f"Resumed from '{cfg.resume_path}' (epoch {saved['epoch']})")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_nll",
            "val_loss",   "val_nll", "val_cindex",
        ])

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        egmdm.train()
        tr_loss = tr_nll = 0.0

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for batch in pbar:
            ct    = batch["ct"][0]
            event = batch["event"][0]
            t_day = batch["survival_time"][0]

            emb    = embed_patient(ct, unet, omnirad, cfg, device).unsqueeze(0).to(device)
            params, reg = egmdm(emb)
            t = (t_day / 365.25).unsqueeze(0).to(device)
            e = event.float().unsqueeze(0).to(device)

            loss, nll = criterion(egmdm, params, reg, t, e)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(egmdm.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item(); tr_nll += nll.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", nll=f"{nll.item():.4f}")

        N = len(train_loader)
        avg_tr_loss, avg_tr_nll = tr_loss / N, tr_nll / N

        # ── Val ────────────────────────────────────────────────────────────
        egmdm.eval()
        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct    = batch["ct"][0]
                event = batch["event"][0]
                t_day = batch["survival_time"][0]

                emb    = embed_patient(ct, unet, omnirad, cfg, device).unsqueeze(0).to(device)
                params, reg = egmdm(emb)
                t = (t_day / 365.25).unsqueeze(0).to(device)
                e = event.float().unsqueeze(0).to(device)

                loss, nll = criterion(egmdm, params, reg, t, e)
                vl_loss += loss.item(); vl_nll += nll.item()

                risk = egmdm.cdf(params, torch.tensor([1.0], device=device)).squeeze().cpu()
                all_risks.append(risk)
                all_times.append(t_day.cpu())
                all_events.append(event.cpu())

        M = len(val_loader)
        avg_vl_loss = vl_loss / M
        avg_vl_nll  = vl_nll  / M
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
            f"C-index: {c_index:.4f}"
        )

        wandb.log({
            "train/lr": lr, "train/loss": avg_tr_loss, "train/nll": avg_tr_nll,
            "val/loss":  avg_vl_loss, "val/nll": avg_vl_nll, "val/cindex": c_index,
            "val/best_loss": min(best_val_loss, avg_vl_loss),
        }, step=epoch)

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
        torch.save(ckpt_payload, cfg.last_ckpt)

        if avg_vl_loss < best_val_loss:
            best_val_loss  = avg_vl_loss
            no_improve_ctr = 0
            torch.save(ckpt_payload, cfg.best_ckpt)
            logger.info(
                f"  ✓ Best model saved → {cfg.best_ckpt}  "
                f"(val_loss={best_val_loss:.4f}  C-index={c_index:.4f})"
            )
            wandb.run.summary.update({
                "best_val_loss":   best_val_loss,
                "best_val_cindex": c_index,
                "best_epoch":      epoch,
            })
        else:
            no_improve_ctr += 1
            logger.info(
                f"  No improvement {no_improve_ctr}/{cfg.early_stop_patience}  "
                f"(best={best_val_loss:.4f})"
            )

        if no_improve_ctr >= cfg.early_stop_patience:
            logger.info(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["early_stop_epoch"] = epoch
            break

    logger.info(f"\nPhase 2 complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {cfg.best_ckpt}")
    wandb.finish()

    return cfg.best_ckpt


if __name__ == "__main__":
    train_survival()