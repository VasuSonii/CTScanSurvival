"""
train_unet.py — Phase 1: 3-D U-Net segmentation
=================================================
Run single experiment:
    python -m kits21.train_unet

Run via launcher (multiple seeds / hyperparams):
    python -m kits21.run_experiments

All hyperparameters live in kits21/configs/unet_config.py.
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.unet_config  import UNetConfig
from data.dataset         import KitsDataset
from losses.seg_loss      import build_seg_loss
from models.unet          import SimpleUNet3D
from utils.inference      import sliding_window_inference
from utils.logging_utils  import log_config, setup_logging
from utils.metrics        import compute_kits_dice
from utils.seed           import set_seed
from utils.wandb_utils    import (
    accumulate_debug_stats, average_debug_stats,
    debug_stats_to_wandb, empty_debug_acc, log_debug_stats,
)


def train_unet(cfg: UNetConfig | None = None) -> str:
    """
    Train one UNet experiment defined by `cfg`.

    Returns
    -------
    best_ckpt_path : str — path to the saved best checkpoint,
                     useful for chaining into Phase 2.
    """
    cfg     = cfg or UNetConfig()
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Seed everything ────────────────────────────────────────────────────
    set_seed(cfg.seed)

    logger, csv_path = setup_logging(cfg.log_dir, prefix="unet")

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_seed{cfg.seed}",
        group   = cfg.experiment_name,   # groups seeds together in W&B UI
        config  = cfg.to_dict(),
        reinit  = True,                  # allow multiple runs per process (launcher)
    )

    logger.info("=" * 60)
    logger.info(f"Phase 1 — UNet Segmentation")
    logger.info(f"W&B run    : {run.url}")
    logger.info(f"Device     : {device}  |  AMP: {use_amp}")
    logger.info("=" * 60)
    log_config(logger, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        rootdir        = cfg.root_dir,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        split_file     = cfg.json_path,
        metadata_path  = os.path.join(cfg.root_dir, "kits.json"),
        p              = cfg.tumour_crop_p,
    )
    train_ds = KitsDataset(**_ds_kwargs, mode="train")
    val_ds   = KitsDataset(**_ds_kwargs, mode="val")

    _dl_kwargs = dict(
        num_workers        = cfg.num_workers,
        pin_memory         = use_amp,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=1,              shuffle=False, **_dl_kwargs)

    logger.info(f"Train: {len(train_ds)} cases | Val: {len(val_ds)} cases")

    # ── Model ──────────────────────────────────────────────────────────────
    raw_model = SimpleUNet3D(
        n_classes     = cfg.num_classes,
        base_channels = cfg.base_channels,
        trilinear     = cfg.trilinear,
    ).to(device)

    # Keep a reference to the raw model for checkpointing.
    # torch.compile() wraps the model and prefixes all state_dict keys with
    # "_orig_mod.", which breaks loading into a plain SimpleUNet3D in Phase 2.
    # Saving raw_model.state_dict() avoids this entirely.
    if hasattr(torch, "compile"):
        model = torch.compile(raw_model)
        logger.info("Model compiled with torch.compile()")
    else:
        model = raw_model

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")
    wandb.log({"model/trainable_params": n_params}, step=0)
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Loss / optimiser / scheduler ───────────────────────────────────────
    criterion = build_seg_loss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    scaler    = GradScaler(device=device.type, enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_val_dice  = -1.0
    no_improve_ctr = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        ckpt = torch.load(cfg.resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_dice  = ckpt.get("val_mean_dice", -1.0)
        no_improve_ctr = ckpt.get("no_improve_ctr", 0)
        logger.info(f"Resumed from '{cfg.resume_path}' (epoch {ckpt['epoch']})")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_ce", "train_dice_loss",
            "train_kidney", "train_tumour", "train_cyst", "train_mean",
            "val_kidney",   "val_tumour",   "val_cyst",   "val_mean",
        ])

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = t_ce = t_dloss = t_kid = t_tum = t_cyst = 0.0
        debug_acc = empty_debug_acc()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for step, batch in enumerate(pbar):
            ct   = batch["ct"].unsqueeze(1).to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(ct)
                loss, ce, dloss, dbg = criterion(logits, mask)

            scaled = loss / cfg.accumulation_steps
            if use_amp:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % cfg.accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            dice = compute_kits_dice(logits.detach(), mask)
            t_loss  += loss.item();  t_ce    += ce.item()
            t_dloss += dloss.item(); t_kid   += dice["kidney_dice"]
            t_tum   += dice["tumour_dice"];  t_cyst += dice["cyst_dice"]
            accumulate_debug_stats(debug_acc, dbg)

            n = step + 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                k=f"{t_kid/n:.3f}", t=f"{t_tum/n:.3f}", c=f"{t_cyst/n:.3f}",
            )

        N       = len(train_loader)
        avg     = lambda s: s / N
        avg_dbg = average_debug_stats(debug_acc, N)
        tr_kid, tr_tum, tr_cyst = avg(t_kid), avg(t_tum), avg(t_cyst)
        tr_mean = (tr_kid + tr_tum + tr_cyst) / 3.0

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        v_kid = v_tum = v_cyst = 0.0

        for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
            volume = batch["ct"][0]
            gt     = batch["mask"][0]
            full_logits = sliding_window_inference(
                model=model, volume=volume, num_classes=cfg.num_classes,
                window=cfg.sw_window, stride=cfg.sw_stride,
                device=device, use_amp=use_amp,
            )
            d = compute_kits_dice(full_logits.unsqueeze(0), gt.unsqueeze(0))
            v_kid += d["kidney_dice"]; v_tum += d["tumour_dice"]; v_cyst += d["cyst_dice"]

        M = len(val_loader)
        vl_kid, vl_tum, vl_cyst = v_kid / M, v_tum / M, v_cyst / M
        vl_mean = (vl_kid + vl_tum + vl_cyst) / 3.0

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────────
        logger.info(
            f"  Train → Loss: {avg(t_loss):.4f}  CE: {avg(t_ce):.4f}  "
            f"Kidney: {tr_kid:.4f}  Tumour: {tr_tum:.4f}  Cyst: {tr_cyst:.4f}  Mean: {tr_mean:.4f}"
        )
        log_debug_stats(logger, "Train", avg_dbg)
        logger.info(
            f"  Val   → Kidney: {vl_kid:.4f}  Tumour: {vl_tum:.4f}  "
            f"Cyst: {vl_cyst:.4f}  Mean: {vl_mean:.4f}"
        )

        wb: dict = {
            "train/lr": lr, "train/loss": avg(t_loss),
            "train/ce_loss": avg(t_ce), "train/dice_loss": avg(t_dloss),
            "train/kidney_dice": tr_kid, "train/tumour_dice": tr_tum,
            "train/cyst_dice": tr_cyst, "train/mean_dice": tr_mean,
            "val/kidney_dice": vl_kid,  "val/tumour_dice": vl_tum,
            "val/cyst_dice": vl_cyst,   "val/mean_dice": vl_mean,
            "val/best_mean_dice": max(best_val_dice, vl_mean),
        }
        wb.update(debug_stats_to_wandb("train", avg_dbg))
        wandb.log(wb, step=epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{avg(t_loss):.6f}", f"{avg(t_ce):.6f}", f"{avg(t_dloss):.6f}",
                f"{tr_kid:.6f}", f"{tr_tum:.6f}", f"{tr_cyst:.6f}", f"{tr_mean:.6f}",
                f"{vl_kid:.6f}", f"{vl_tum:.6f}", f"{vl_cyst:.6f}", f"{vl_mean:.6f}",
            ])

        # ── Checkpoint — written to cfg.run_dir, never overwritten by other runs
        ckpt_payload = {
            "epoch":            epoch,
            "model_state":      raw_model.state_dict(),   # never has _orig_mod. prefix
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "val_mean_dice":    vl_mean,
            "no_improve_ctr":   no_improve_ctr,
            # Save full config for traceability
            "config":           cfg.to_dict(),
        }
        torch.save(ckpt_payload, cfg.last_ckpt)

        if vl_mean > best_val_dice:
            best_val_dice  = vl_mean
            no_improve_ctr = 0
            torch.save(ckpt_payload, cfg.best_ckpt)
            logger.info(f"  ✓ Best model saved → {cfg.best_ckpt}  (mean_dice={best_val_dice:.4f})")
            wandb.run.summary.update({
                "best_val_mean_dice": best_val_dice,
                "best_epoch":         epoch,
            })
        else:
            no_improve_ctr += 1
            logger.info(
                f"  No improvement {no_improve_ctr}/{cfg.early_stop_patience}  "
                f"(best={best_val_dice:.4f})"
            )

        if no_improve_ctr >= cfg.early_stop_patience:
            logger.info(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["early_stop_epoch"] = epoch
            break

    logger.info(f"\nPhase 1 complete. Best val mean Dice: {best_val_dice:.4f}")
    logger.info(f"Best checkpoint: {cfg.best_ckpt}")
    wandb.finish()

    return cfg.best_ckpt


if __name__ == "__main__":
    train_unet()