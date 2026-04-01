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

Mask source (cfg.use_gt_mask)
------------------------------
  False (default) — frozen UNet predicts the mask via sliding-window inference.
                    This is the realistic deployment setting.
  True            — ground-truth segmentation mask is taken directly from the
                    dataset.  Use this as an upper-bound experiment to measure
                    EGMDM performance free of UNet prediction error.

Slice pooling (cfg.slice_pooling)
----------------------------------
  "mean"      — unweighted mean over slice embeddings (no extra parameters).
  "attention" — gated attention pooling (Ilse et al. 2018); trains a small
                attention network jointly with the EGMDM head.  The attention
                weights are logged to W&B for interpretability.

Early stopping
--------------
Tracks val C-index (higher = better).  Best checkpoint is saved whenever
C-index improves.
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
os.environ["WANDB_MODE"] = "offline"
from configs.survival_config import SurvivalConfig
from data.dataset            import KitsDataset
from losses.survival_loss    import EGMDMLoss
from models.egmdm            import EGMDMHead
from models.omnirad          import GatedAttentionPooling, OmniRadEncoder
from models.unet             import SimpleUNet3D
from utils.inference         import sliding_window_predict
from utils.logging_utils     import log_config, setup_logging
from utils.metrics           import concordance_index
from utils.seed              import set_seed
from data.clinical_preprocessor import ClinicalPreprocessor
from models.clinical_mlp        import ClinicalMLP


# ─── Patient embedding pipeline ───────────────────────────────────────────────

@torch.no_grad()
def embed_patient(
    ct:       torch.Tensor,                       # (D, H, W) float32 on CPU
    unet:     torch.nn.Module,
    omnirad:  OmniRadEncoder,
    pooling:  torch.nn.Module,                    # GatedAttentionPooling or identity
    cfg:      SurvivalConfig,
    device:   torch.device,
    gt_mask:  torch.Tensor | None = None,         # (D, H, W) int64 on CPU
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build a patient-level embedding vector from a CT volume.

    Pipeline
    --------
    1. Obtain segmentation mask:
         - cfg.use_gt_mask=True  → use gt_mask directly (must be provided)
         - cfg.use_gt_mask=False → run frozen UNet in sliding-window mode
    2. OmniRad encodes every (CT slice, mask slice) pair → (D, embed_dim) on CPU.
    3. Pool across depth with the chosen pooling module:
         - "mean"      → unweighted mean → (embed_dim,) on CPU
         - "attention" → gated attention → (embed_dim,) on CPU
                         also returns (D,) attention weights for logging

    Parameters
    ----------
    ct      : (D, H, W) CT volume normalised to [0, 1], on CPU
    unet    : frozen SimpleUNet3D — only used when cfg.use_gt_mask is False
    omnirad : frozen OmniRadEncoder
    pooling : GatedAttentionPooling (trainable) when cfg.slice_pooling="attention",
              or torch.nn.Identity (passthrough) when cfg.slice_pooling="mean"
    cfg     : SurvivalConfig — controls use_gt_mask, slice_pooling, etc.
    device  : GPU/CPU device for UNet inference and attention
    gt_mask : (D, H, W) int64 ground-truth mask from the dataset batch;
              required when cfg.use_gt_mask is True, ignored otherwise

    Returns
    -------
    embedding      : (embed_dim,) tensor on CPU
    attn_weights   : (D,) tensor on CPU if slice_pooling="attention", else None
    """
    # ── Step 1: mask ──────────────────────────────────────────────────────
    if cfg.use_gt_mask:
        if gt_mask is None:
            raise ValueError("cfg.use_gt_mask is True but gt_mask was not provided.")
        mask = gt_mask   # (D, H, W) int64 on CPU
    else:
        mask = sliding_window_predict(
            model       = unet,
            volume      = ct,
            num_classes = cfg.num_classes,
            window      = cfg.sw_window,
            stride      = cfg.sw_stride,
            device      = device,
        )   # (D, H, W) int64 on CPU

    # ── Step 2: per-slice OmniRad embeddings ──────────────────────────────
    slice_embs = omnirad.encode_volume(
        ct          = ct,
        mask        = mask,
        num_classes = cfg.num_classes,
        batch_size  = cfg.omni_batch,
    )   # (D, embed_dim) on CPU

    # ── Step 3: pool across depth ─────────────────────────────────────────
    if cfg.slice_pooling == "attention":
        # Attention is trainable — run with grad enabled even though this
        # function is decorated @no_grad; we re-enable it explicitly here.
        with torch.enable_grad():
            emb, attn_weights = pooling(slice_embs.to(device))
        return emb.cpu(), attn_weights.detach().cpu()
    else:
        return slice_embs.mean(dim=0), None   # (embed_dim,) on CPU


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_survival(cfg: SurvivalConfig | None = None) -> str:
    """
    Train one survival experiment defined by `cfg`.

    Returns
    -------
    best_ckpt_path : str
    """
    cfg    = cfg or SurvivalConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)

    logger, csv_path = setup_logging(cfg.log_dir, prefix="survival")

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_{cfg._variant}_seed{cfg.seed}",
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        notes   = cfg.wandb_notes or None,
        tags    = cfg.wandb_tags  or [],
        reinit  = True,
    )

    logger.info("=" * 60)
    logger.info(f"Phase 2 — EGMDM Survival Analysis")
    logger.info(f"W&B run     : {run.url}")
    logger.info(f"Device      : {device}")
    logger.info("=" * 60)
    log_config(logger, cfg)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        rootdir        = cfg.root_dir,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        split_file     = cfg.json_path,
        metadata_path  = os.path.join(cfg.root_dir, "kits23.json"),
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

    # ── Imaging pipeline (conditional on cfg.use_imaging) ──────────────────
    if cfg.use_imaging:
        if not os.path.exists(cfg.unet_ckpt):
            raise FileNotFoundError(
                f"UNet checkpoint not found: '{cfg.unet_ckpt}'\n"
                "Run Phase 1 (train_unet) first, or set cfg.unet_ckpt correctly."
            )
        unet = SimpleUNet3D(
            n_classes     = cfg.num_classes,
            base_channels = cfg.unet_base_channels,
            trilinear     = cfg.unet_trilinear,
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

        omnirad = OmniRadEncoder(device=device, frozen=True)
        logger.info("OmniRad loaded and frozen.")

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
    else:
        unet    = None
        omnirad = None
        pooling = None

    # ── Clinical MLP (conditional on cfg.use_clinical) ───────────────────
    clinical_feats_train: dict = {}
    clinical_feats_val:   dict = {}
    clinical_mlp = None

    if cfg.use_clinical:
        metadata_path = os.path.join(cfg.root_dir, "kits23.json")
        cp = ClinicalPreprocessor(
            metadata_path     = metadata_path,
            missing_threshold = cfg.missing_threshold,
        )
        clinical_feats_train, clinical_feats_val, clin_dim = cp.fit_transform(
            train_ids = train_ds.cases,
            val_ids   = val_ds.cases,
        )
        cp.save(cfg.clinical_preprocessor_path)
        logger.info(
            f"Clinical MLP: input_dim={clin_dim}  "
            f"output_dim={cfg.clinical_dim}"
        )
        clinical_mlp = ClinicalMLP(
            input_dim   = clin_dim,
            output_dim  = cfg.clinical_dim,
            hidden_dims = cfg.clinical_hidden_dims,
            dropout     = cfg.clinical_dropout,
        ).to(device)
        n_clin = sum(p.numel() for p in clinical_mlp.parameters())
        logger.info(f"ClinicalMLP            params: {n_clin:,}")
        wandb.log({"model/clinical_mlp_params": n_clin}, step=0)

    # ── EGMDM Head ─────────────────────────────────────────────────────────
    egmdm = EGMDMHead(
        input_size  = cfg.egmdm_input_dim,  # embed_dim [+ clinical_dim]
        hidden_size = cfg.egmdm_hidden_size,
        E           = cfg.egmdm_E,
        K           = cfg.egmdm_K,
        dropout     = cfg.egmdm_dropout,
    ).to(device)

    n_egmdm = sum(p.numel() for p in egmdm.parameters() if p.requires_grad)
    logger.info(f"EGMDMHead              params: {n_egmdm:,}")
    wandb.log({"model/egmdm_params": n_egmdm}, step=0)

    # ── Loss / optimiser / scheduler ───────────────────────────────────────
    # Optimise pooling + EGMDM jointly; OmniRad and UNet are frozen.
    trainable_params = list(egmdm.parameters())
    if pooling is not None:
        trainable_params += list(pooling.parameters())
    if clinical_mlp is not None:
        trainable_params += list(clinical_mlp.parameters())

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
        if clinical_mlp is not None and "clinical_mlp_state" in saved:
            clinical_mlp.load_state_dict(saved["clinical_mlp_state"])
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
        if clinical_mlp is not None:
            clinical_mlp.train()
        tr_loss = tr_nll = 0.0

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for batch in pbar:
            ct      = batch["ct"][0]      # (D, H, W) CPU
            gt_mask = batch["mask"][0]    # (D, H, W) CPU int64
            event   = batch["event"][0]
            t_day   = batch["survival_time"][0]

            parts = []
            if cfg.use_imaging:
                img_emb, _ = embed_patient(
                    ct      = ct,
                    unet    = unet,
                    omnirad = omnirad,
                    pooling = pooling,
                    cfg     = cfg,
                    device  = device,
                    gt_mask = gt_mask,
                )
                parts.append(img_emb)                    # (embed_dim,)
            if clinical_mlp is not None:
                cid      = batch["caseid"][0]
                clin_vec = clinical_feats_train.get(cid)
                if clin_vec is not None:
                    parts.append(
                        clinical_mlp(clin_vec.to(device)).cpu()
                    )                                    # (clinical_dim,)
            emb = torch.cat(parts, dim=0).unsqueeze(0).to(device)  # (1, egmdm_input_dim)

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
        if clinical_mlp is not None:
            clinical_mlp.eval()
        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []
        # Collect attention weights from the last val epoch for W&B logging
        all_attn: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct      = batch["ct"][0]
                gt_mask = batch["mask"][0]
                event   = batch["event"][0]
                t_day   = batch["survival_time"][0]

                parts   = []
                attn_w  = None
                if cfg.use_imaging:
                    img_emb, attn_w = embed_patient(
                        ct      = ct,
                        unet    = unet,
                        omnirad = omnirad,
                        pooling = pooling,
                        cfg     = cfg,
                        device  = device,
                        gt_mask = gt_mask,
                    )
                    parts.append(img_emb)
                if clinical_mlp is not None:
                    cid      = batch["caseid"][0]
                    clin_vec = clinical_feats_val.get(cid)
                    if clin_vec is not None:
                        parts.append(
                            clinical_mlp(clin_vec.to(device)).cpu()
                        )
                emb = torch.cat(parts, dim=0).unsqueeze(0).to(device)

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
        # Log mean attention weight entropy as a measure of how focused
        # the model's attention is across slices.  Each patient has a
        # different D so we compute entropy per-patient then average —
        # torch.stack would fail because the tensors have unequal lengths.
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
        if clinical_mlp is not None:
            ckpt_payload["clinical_mlp_state"] = clinical_mlp.state_dict()

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

    logger.info(f"\nPhase 2 complete. Best val C-index: {best_cindex:.4f}")
    logger.info(f"Best checkpoint: {cfg.best_ckpt}")
    wandb.finish()

    return cfg.best_ckpt


if __name__ == "__main__":
    train_survival()