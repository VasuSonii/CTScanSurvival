"""
train_unet_hector.py — Phase 1: 2-channel UNet segmentation on HECKTOR Task 1
==============================================================================
Run:
    python train_unet_hector.py

Trains a SimpleUNet3D with in_channels=2 (CT + PT) to segment head & neck
tumours.  The best checkpoint is later used frozen in train_survival_hector.py
to predict masks for Task 2 patients.

Input to UNet : (B, 2, D, H, W) — CT and PT stacked on channel dim
Output        : (B, num_classes, D, H, W) logits
"""

import csv
import os

import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.hector_unet_config import HectorUNetConfig
from data.dataset_hector        import HectorTask1Dataset
from losses.seg_loss            import build_seg_loss
from models.unet                import SimpleUNet3D
from utils.inference            import sliding_window_predict
from utils.logging_utils        import log_config, setup_logging
from utils.metrics              import compute_hector_dice
from utils.seed                 import set_seed


def train_unet_hector(cfg: HectorUNetConfig | None = None) -> str:
    cfg    = cfg or HectorUNetConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    logger, csv_path = setup_logging(cfg.log_dir, prefix="unet_hector")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_seed{cfg.seed}",
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        reinit  = True,
    )

    logger.info("=" * 60)
    logger.info("HECKTOR Phase 1 — 2-channel UNet Segmentation")
    logger.info(f"W&B run  : {run.url}")
    logger.info(f"Device   : {device}")
    logger.info("=" * 60)
    log_config(logger, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        task1_dir      = cfg.task1_dir,
        split_file     = cfg.split_file,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        p              = cfg.tumour_crop_p,
        crop_depth     = cfg.crop_depth,
    )
    train_ds = HectorTask1Dataset(**_ds_kwargs, mode="train")
    val_ds   = HectorTask1Dataset(**_ds_kwargs, mode="val")

    _dl_kwargs = dict(
        batch_size         = cfg.batch_size,
        num_workers        = cfg.num_workers,
        pin_memory         = True,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,
                              batch_size=1, num_workers=cfg.num_workers,
                              persistent_workers=cfg.num_workers > 0,
                              prefetch_factor=2 if cfg.num_workers > 0 else None)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────
    raw_model = SimpleUNet3D(
        n_classes     = cfg.num_classes,
        base_channels = cfg.base_channels,
        trilinear     = cfg.trilinear,
        in_channels   = cfg.in_channels,   # 2 for CT+PT
    ).to(device)
    model = torch.compile(raw_model)

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    logger.info(f"UNet params: {n_params:,}")

    # ── Loss / optimiser ───────────────────────────────────────────────────
    criterion = build_seg_loss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6
    )
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_dice = 0.0
    no_improve    = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        ckpt = torch.load(cfg.resume_path, map_location=device)
        raw_model.load_state_dict({
            k.removeprefix("_orig_mod."): v
            for k, v in ckpt["model_state"].items()
        })
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("val_mean_dice", 0.0)
        no_improve    = ckpt.get("no_improve_ctr", 0)
        logger.info(f"Resumed from '{cfg.resume_path}' (epoch {ckpt['epoch']})")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "train_loss", "train_ce", "train_seg",
            "train_tumour_fp", "train_tumour_fn",
            "train_lymph_fp",  "train_lymph_fn",
            "val_tumour_dice", "val_tumour_recall", "val_tumour_precision",
            "val_tumour_fp",   "val_tumour_fn",
            "val_lymph_dice",  "val_lymph_recall",  "val_lymph_precision",
            "val_lymph_fp",    "val_lymph_fn",
            "val_mean_fg_dice",
        ])

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_ce = tr_seg = 0.0
        # Patient-level FP/FN accumulated from loss debug_stats
        tr_fp = {1: 0.0, 2: 0.0}   # class 1=tumour, 2=lymph
        tr_fn = {1: 0.0, 2: 0.0}
        tr_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for step, batch in enumerate(pbar, 1):
            ct_pt = batch["ct_pt"].to(device)   # (B, D, 2, H, W)
            mask  = batch["mask"].to(device)     # (B, D, H, W)

            # UNet expects (B, C, D, H, W) — permute channel to dim 1
            ct_pt = ct_pt.permute(0, 2, 1, 3, 4)  # (B, 2, D, H, W)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits                  = model(ct_pt)
                loss, ce, dloss, dbg    = criterion(logits, mask)
                loss                    = loss / cfg.accumulation_steps

            scaler.scale(loss).backward()

            if step % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tr_loss += loss.item() * cfg.accumulation_steps
            tr_ce   += ce.item()
            tr_seg  += dloss.item()
            # Accumulate per-class FP/FN from debug stats (patient-level)
            if 'false_pos' in dbg and len(dbg['false_pos']) >= 3:
                tr_fp[1] += dbg['false_pos'][1]; tr_fp[2] += dbg['false_pos'][2]
                tr_fn[1] += dbg['false_neg'][1]; tr_fn[2] += dbg['false_neg'][2]
            tr_steps += 1
            pbar.set_postfix(loss=f"{loss.item() * cfg.accumulation_steps:.4f}")

        n = len(train_loader)
        avg_tr_loss = tr_loss / n
        avg_tr_ce   = tr_ce   / n
        avg_tr_seg  = tr_seg  / n
        avg_tr_fp   = {k: v / max(tr_steps, 1) for k, v in tr_fp.items()}
        avg_tr_fn   = {k: v / max(tr_steps, 1) for k, v in tr_fn.items()}

        # ── Val ────────────────────────────────────────────────────────────
        raw_model.eval()
        per_case: list[dict] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct      = batch["ct_pt"][0, :, 0]   # (D, H, W) CT channel
                pt      = batch["ct_pt"][0, :, 1]   # (D, H, W) PT channel
                gt_mask = batch["mask"][0]

                ct_pt_vol = torch.stack([ct, pt], dim=1)  # (D, 2, H, W)

                pred_mask = sliding_window_predict(
                    model       = raw_model,
                    volume      = ct_pt_vol,
                    num_classes = cfg.num_classes,
                    window      = cfg.sw_window,
                    stride      = cfg.sw_stride,
                    device      = device,
                )
                per_case.append(compute_hector_dice(pred_mask, gt_mask))

        # Average each metric across all val cases
        val_metrics = {
            k: sum(d[k] for d in per_case) / len(per_case)
            for k in per_case[0]
        }
        val_mean_dice = val_metrics["mean_fg_dice"]
        scheduler.step()

        logger.info(
            f"  Train  loss={avg_tr_loss:.4f}  ce={avg_tr_ce:.4f}  "
            f"seg={avg_tr_seg:.4f}"
        )
        logger.info(
            f"  Train  tumour FP={avg_tr_fp[1]:.0f}  FN={avg_tr_fn[1]:.0f}  "
            f"lymph FP={avg_tr_fp[2]:.0f}  FN={avg_tr_fn[2]:.0f}"
        )
        logger.info(
            f"  Val    tumour dice={val_metrics['tumour_dice']:.4f}  "
            f"recall={val_metrics['tumour_recall']:.4f}  "
            f"prec={val_metrics['tumour_precision']:.4f}  "
            f"FP={val_metrics['tumour_fp']:.0f}  FN={val_metrics['tumour_fn']:.0f}"
        )
        logger.info(
            f"  Val    lymph  dice={val_metrics['lymph_dice']:.4f}  "
            f"recall={val_metrics['lymph_recall']:.4f}  "
            f"prec={val_metrics['lymph_precision']:.4f}  "
            f"FP={val_metrics['lymph_fp']:.0f}  FN={val_metrics['lymph_fn']:.0f}"
        )
        logger.info(f"  Val    mean FG dice={val_mean_dice:.4f}")
        wandb.log({
            "train/lr":                lr,
            "train/loss":              avg_tr_loss,
            "train/ce_loss":           avg_tr_ce,
            "train/seg_loss":          avg_tr_seg,
            "train/tumour_fp":         avg_tr_fp[1],
            "train/tumour_fn":         avg_tr_fn[1],
            "train/lymph_fp":          avg_tr_fp[2],
            "train/lymph_fn":          avg_tr_fn[2],
            "val/tumour_dice":         val_metrics["tumour_dice"],
            "val/tumour_recall":       val_metrics["tumour_recall"],
            "val/tumour_precision":    val_metrics["tumour_precision"],
            "val/tumour_fp":           val_metrics["tumour_fp"],
            "val/tumour_fn":           val_metrics["tumour_fn"],
            "val/lymph_dice":          val_metrics["lymph_dice"],
            "val/lymph_recall":        val_metrics["lymph_recall"],
            "val/lymph_precision":     val_metrics["lymph_precision"],
            "val/lymph_fp":            val_metrics["lymph_fp"],
            "val/lymph_fn":            val_metrics["lymph_fn"],
            "val/mean_fg_dice":        val_mean_dice,
        }, step=epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{avg_tr_loss:.6f}", f"{avg_tr_ce:.6f}", f"{avg_tr_seg:.6f}",
                f"{avg_tr_fp[1]:.0f}", f"{avg_tr_fn[1]:.0f}",
                f"{avg_tr_fp[2]:.0f}", f"{avg_tr_fn[2]:.0f}",
                f"{val_metrics['tumour_dice']:.4f}",
                f"{val_metrics['tumour_recall']:.4f}",
                f"{val_metrics['tumour_precision']:.4f}",
                f"{val_metrics['tumour_fp']:.0f}",
                f"{val_metrics['tumour_fn']:.0f}",
                f"{val_metrics['lymph_dice']:.4f}",
                f"{val_metrics['lymph_recall']:.4f}",
                f"{val_metrics['lymph_precision']:.4f}",
                f"{val_metrics['lymph_fp']:.0f}",
                f"{val_metrics['lymph_fn']:.0f}",
                f"{val_mean_dice:.4f}",
            ])

        ckpt = {
            "epoch":            epoch,
            "model_state":      raw_model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "val_mean_dice":    val_mean_dice,
            "val_tumour_dice":  val_metrics["tumour_dice"],
            "val_lymph_dice":   val_metrics["lymph_dice"],
            "no_improve_ctr":   no_improve,
            "config":           cfg.to_dict(),
        }
        torch.save(ckpt, cfg.last_ckpt)

        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            no_improve    = 0
            torch.save(ckpt, cfg.best_ckpt)
            logger.info(
                f"  ✓ Best saved  tumour={val_metrics['tumour_dice']:.4f}  "
                f"lymph={val_metrics['lymph_dice']:.4f}  "
                f"mean={val_mean_dice:.4f}"
            )
            wandb.run.summary.update({
                "best_val_mean_dice":   val_mean_dice,
                "best_val_tumour_dice": val_metrics["tumour_dice"],
                "best_val_lymph_dice":  val_metrics["lymph_dice"],
                "best_epoch":           epoch,
            })
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}.")
                break

    logger.info(f"\nDone. Best val mean FG dice: {best_val_dice:.4f}")
    wandb.finish()
    return cfg.best_ckpt


if __name__ == "__main__":
    train_unet_hector()