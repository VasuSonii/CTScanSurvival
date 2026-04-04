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
def get_slice_embeddings(
    ct:       torch.Tensor,               # (D, H, W) float32 on CPU
    unet:     torch.nn.Module,
    omnirad:  OmniRadEncoder,
    cfg:      SurvivalConfig,
    device:   torch.device,
    gt_mask:  torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Steps 1-2 only: mask prediction + OmniRad encoding.
    Returns slice_embs (D, embed_dim) on CPU, fully detached.

    Pooling is intentionally NOT done here so that GatedAttentionPooling
    runs inside the training loop where the computation graph is live.
    Previously pooling was called inside @no_grad then moved to CPU which
    broke the gradient path: emb.cpu() detaches from the graph, so
    pooling received zero gradients despite torch.enable_grad().
    """
    if cfg.use_gt_mask:
        if gt_mask is None:
            raise ValueError("cfg.use_gt_mask is True but gt_mask was not provided.")
        mask = gt_mask
    else:
        mask = sliding_window_predict(
            model       = unet,
            volume      = ct,
            num_classes = cfg.num_classes,
            window      = cfg.sw_window,
            stride      = cfg.sw_stride,
            device      = device,
        )

    return omnirad.encode_volume(
        ct          = ct,
        mask        = mask,
        num_classes = cfg.num_classes,
        batch_size  = cfg.omni_batch,
    )   # (D, embed_dim) on CPU, detached


def pool_slice_embeddings(
    slice_embs: torch.Tensor,            # (D, embed_dim) on CPU
    pooling:    torch.nn.Module | None,  # GatedAttentionPooling or None
    cfg:        SurvivalConfig,
    device:     torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Step 3: pool (D, embed_dim) → (embed_dim,) on GPU, keeping grad alive.

    Called inside the training loop (grad context active) so gradients
    flow correctly back through GatedAttentionPooling.
    The returned embedding stays on GPU — do NOT move to CPU before backward.

    Returns
    -------
    emb          : (embed_dim,) on device  ← stays on GPU for backward
    attn_weights : (D,) on CPU or None
    """
    embs_gpu = slice_embs.to(device)   # (D, embed_dim) on GPU

    if cfg.slice_pooling == "attention" and pooling is not None:
        emb, attn_weights = pooling(embs_gpu)   # (embed_dim,) on GPU, grad alive
        return emb, attn_weights.detach().cpu()
    else:
        return embs_gpu.mean(dim=0), None       # (embed_dim,) on GPU


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
        tr_loss = tr_nll = tr_reg_div = tr_reg_ent = 0.0
        # Diagnostic accumulators
        tr_img_norm  = tr_clin_norm  = 0.0   # embedding L2 norms
        tr_grad_egmdm = tr_grad_pool = tr_grad_clin = 0.0
        tr_n = 0

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for batch in pbar:
            ct      = batch["ct"][0]      # (D, H, W) CPU
            gt_mask = batch["mask"][0]    # (D, H, W) CPU int64
            event   = batch["event"][0]
            t_day   = batch["survival_time"][0]

            parts = []   # all tensors stay on GPU for backward
            if cfg.use_imaging:
                # Step 1+2: frozen (no_grad), returns CPU tensor
                slice_embs = get_slice_embeddings(
                    ct=ct, unet=unet, omnirad=omnirad,
                    cfg=cfg, device=device, gt_mask=gt_mask,
                )
                # Step 3: pooling runs with grad, stays on GPU
                img_emb, _ = pool_slice_embeddings(
                    slice_embs=slice_embs, pooling=pooling,
                    cfg=cfg, device=device,
                )
                # L2-normalise so imaging and clinical live on same scale
                img_emb_n = torch.nn.functional.normalize(img_emb, dim=0)
                parts.append(img_emb_n)
                tr_img_norm += img_emb.norm().item()   # log pre-norm for diagnosis
            if clinical_mlp is not None:
                cid      = batch["caseid"][0]
                clin_vec = clinical_feats_train.get(cid)
                if clin_vec is not None:
                    clin_emb = clinical_mlp(clin_vec.to(device))  # on GPU
                    clin_emb_n = torch.nn.functional.normalize(clin_emb, dim=0)
                    parts.append(clin_emb_n)
                    tr_clin_norm += clin_emb.norm().item()
            emb = torch.cat(parts, dim=0).unsqueeze(0)  # (1, egmdm_input_dim) on GPU

            params, reg = egmdm(emb)
            t = (t_day / 365.25).unsqueeze(0).to(device)
            e = event.float().unsqueeze(0).to(device)

            loss, nll = criterion(egmdm, params, reg, t, e)
            optimizer.zero_grad()
            loss.backward()

            # Collect per-module gradient norms BEFORE clipping
            def _grad_norm(params_iter):
                g = [p.grad.detach().norm() for p in params_iter if p.grad is not None]
                return torch.stack(g).norm().item() if g else 0.0
            tr_grad_egmdm += _grad_norm(egmdm.parameters())
            if pooling is not None:
                tr_grad_pool  += _grad_norm(pooling.parameters())
            if clinical_mlp is not None:
                tr_grad_clin  += _grad_norm(clinical_mlp.parameters())
            tr_n += 1

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_nll  += nll.item()
            # Track reg components separately
            tr_reg_div += reg.get('L_div', torch.tensor(0.0)).item()
            tr_reg_ent += reg.get('L_ent', torch.tensor(0.0)).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", nll=f"{nll.item():.4f}")

        n = max(tr_n, 1)
        avg_tr_loss      = tr_loss      / len(train_loader)
        avg_tr_nll       = tr_nll       / len(train_loader)
        avg_tr_reg_div   = tr_reg_div   / len(train_loader)
        avg_tr_reg_ent   = tr_reg_ent   / len(train_loader)
        avg_img_norm     = tr_img_norm  / n
        avg_clin_norm    = tr_clin_norm / n
        avg_grad_egmdm   = tr_grad_egmdm / n
        avg_grad_pool    = tr_grad_pool  / n
        avg_grad_clin    = tr_grad_clin  / n

        # ── Val ────────────────────────────────────────────────────────────
        egmdm.eval()
        if pooling is not None:
            pooling.eval()
        if clinical_mlp is not None:
            clinical_mlp.eval()
        vl_loss = vl_nll = 0.0
        all_risks, all_times, all_events = [], [], []
        all_attn:        list[torch.Tensor] = []
        all_img_norms:   list[float]        = []
        all_clin_norms:  list[float]        = []
        all_sigma_mean:  list[float]        = []
        all_sigma_min:   list[float]        = []
        all_mix_entropy: list[float]        = []
        all_attn_max:    list[float]        = []
        all_reg_div:     list[float]        = []
        all_reg_ent:     list[float]        = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct      = batch["ct"][0]
                gt_mask = batch["mask"][0]
                event   = batch["event"][0]
                t_day   = batch["survival_time"][0]

                parts   = []
                attn_w  = None
                _img_n = _clin_n = 0.0
                if cfg.use_imaging:
                    slice_embs = get_slice_embeddings(
                        ct=ct, unet=unet, omnirad=omnirad,
                        cfg=cfg, device=device, gt_mask=gt_mask,
                    )
                    img_emb, attn_w = pool_slice_embeddings(
                        slice_embs=slice_embs, pooling=pooling,
                        cfg=cfg, device=device,
                    )
                    _img_n = img_emb.norm().item()
                    img_emb_n = torch.nn.functional.normalize(img_emb, dim=0)
                    parts.append(img_emb_n)
                if clinical_mlp is not None:
                    cid      = batch["caseid"][0]
                    clin_vec = clinical_feats_val.get(cid)
                    if clin_vec is not None:
                        clin_emb   = clinical_mlp(clin_vec.to(device))
                        _clin_n    = clin_emb.norm().item()
                        clin_emb_n = torch.nn.functional.normalize(clin_emb, dim=0)
                        parts.append(clin_emb_n)
                all_img_norms.append(_img_n)
                all_clin_norms.append(_clin_n)
                emb = torch.cat(parts, dim=0).unsqueeze(0)  # on GPU

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
                # Sigma diagnostics — collapse to 0 = memorisation
                all_sigma_mean.append(params['sigma'].mean().item())
                all_sigma_min.append(params['sigma'].min().item())
                # Mixture entropy — low = model is overconfident
                w = params['w'].clamp(1e-8, 1.0)
                mix_ent = -(w * w.log()).sum(-1).mean().item()
                all_mix_entropy.append(mix_ent)
                # Attention saturation — max weight → 1 means collapsed
                if attn_w is not None:
                    all_attn_max.append(attn_w.max().item())
                # Reg loss components
                _, reg_v = egmdm(emb)
                all_reg_div.append(reg_v.get('L_div', torch.tensor(0.0)).item())
                all_reg_ent.append(reg_v.get('L_ent', torch.tensor(0.0)).item())

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
            "train/lr":              lr,
            "train/loss":            avg_tr_loss,
            "train/nll":             avg_tr_nll,
            # Embedding norm — high imaging vs low clinical → imaging dominates
            "train/img_emb_norm":    avg_img_norm,
            "train/clin_emb_norm":   avg_clin_norm,
            # Gradient norms — tells which module is actually learning
            "train/grad_egmdm":      avg_grad_egmdm,
            "train/grad_pooling":    avg_grad_pool,
            "train/grad_clinical":   avg_grad_clin,
            "val/loss":              avg_vl_loss,
            "val/nll":               avg_vl_nll,
            "val/cindex":            c_index,
            "val/best_cindex":       max(best_cindex, c_index),
            # Val embedding norms
            "val/img_emb_norm":      sum(all_img_norms)  / max(len(all_img_norms),  1),
            "val/clin_emb_norm":     sum(all_clin_norms) / max(len(all_clin_norms), 1),
            # EGMDM distribution diagnostics
            "val/sigma_mean":        sum(all_sigma_mean)  / max(len(all_sigma_mean),  1),
            "val/sigma_min":         sum(all_sigma_min)   / max(len(all_sigma_min),   1),
            "val/mixture_entropy":   sum(all_mix_entropy) / max(len(all_mix_entropy), 1),
            # Attention saturation
            "val/attn_max_weight":   sum(all_attn_max) / max(len(all_attn_max), 1) if all_attn_max else 0.0,
            # Reg loss components
            "train/reg_div":         avg_tr_reg_div,
            "train/reg_ent":         avg_tr_reg_ent,
            "val/reg_div":           sum(all_reg_div) / max(len(all_reg_div), 1),
            "val/reg_ent":           sum(all_reg_ent) / max(len(all_reg_ent), 1),
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