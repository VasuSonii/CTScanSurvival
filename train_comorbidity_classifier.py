"""
train_comorbidity_classifier.py
================================
Probe whether OmniRad CT image features can predict comorbidities.

Pipeline
--------
  KitsDataset (CT + GT mask, full volume)
    → OmniRad.encode_volume()  [frozen]     → (D, 768)
    → GatedAttentionPooling    [trainable]  → (768,)
    → ComorbidityClassifier    [trainable]  → (5,) logits
    → BCEWithLogitsLoss per label with pos_weight for class imbalance

Metrics logged per label and aggregate:
  Train : loss per label, total loss, gradient norms
  Val   : AUC-ROC, AUC-PR, F1, precision, recall, accuracy per label
          + macro averages across all 5 labels
          + W&B confusion matrix per label at best epoch

Labels
------
  0  chronic_kidney_disease
  1  mild_liver_disease
  2  moderate_to_severe_liver_disease
  3  diabetes_mellitus_with_end_organ_damage
  4  peripheral_vascular_disease

Run:
    python train_comorbidity_classifier.py
"""

import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from configs.comorbidity_config   import ComorbidityConfig
from data.comorbidity_labels      import (
    COMORBIDITY_LABELS, N_LABELS, load_comorbidity_labels
)
from data.dataset                 import KitsDataset
from models.comorbidity_classifier import ComorbidityClassifier
from models.omnirad               import GatedAttentionPooling, OmniRadEncoder
from utils.inference              import sliding_window_predict
from utils.logging_utils          import setup_logging
from utils.seed                   import set_seed


# ─── Embedding helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def get_slice_embeddings(
    ct:      torch.Tensor,     # (D, H, W)
    mask:    torch.Tensor,     # (D, H, W)
    omnirad: OmniRadEncoder,
    cfg:     ComorbidityConfig,
) -> torch.Tensor:
    """Frozen OmniRad → (D, 768) on CPU."""
    return omnirad.encode_volume(
        ct          = ct,
        mask        = mask,
        num_classes = 4,       # KiTS: background, kidney, tumour, cyst
        batch_size  = cfg.omni_batch,
    )


def pool_embeddings(
    slice_embs: torch.Tensor,           # (D, 768) CPU
    pooling:    GatedAttentionPooling,
    device:     torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool slice embeddings → (768,) on GPU, grad alive.
    Returns (embedding, attn_weights).
    """
    emb, attn_w = pooling(slice_embs.to(device))
    return emb, attn_w.detach().cpu()


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_label_metrics(
    logits_all: torch.Tensor,   # (N, 5) float
    labels_all: torch.Tensor,   # (N, 5) float 0/1
    threshold:  float = 0.5,
) -> dict:
    """
    Per-label and macro-averaged metrics.
    Returns flat dict suitable for wandb.log.
    """
    probs = torch.sigmoid(logits_all).numpy()
    y     = labels_all.numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {}

    aucs_roc = []
    aucs_pr  = []
    f1s      = []

    for i, label in enumerate(COMORBIDITY_LABELS):
        short = label.replace("_", " ")

        # AUC-ROC — needs both classes present; skip if only one class in val
        if y[:, i].sum() > 0 and (1 - y[:, i]).sum() > 0:
            auc_roc = roc_auc_score(y[:, i], probs[:, i])
            auc_pr  = average_precision_score(y[:, i], probs[:, i])
        else:
            auc_roc = float("nan")
            auc_pr  = float("nan")

        f1   = f1_score(y[:, i], preds[:, i], zero_division=0)
        prec = precision_score(y[:, i], preds[:, i], zero_division=0)
        rec  = recall_score(y[:, i], preds[:, i], zero_division=0)
        acc  = (preds[:, i] == y[:, i]).mean()

        n_pos = int(y[:, i].sum())
        n_neg = int((1 - y[:, i]).sum())

        metrics[f"val/{label}/auc_roc"]   = auc_roc
        metrics[f"val/{label}/auc_pr"]    = auc_pr
        metrics[f"val/{label}/f1"]        = f1
        metrics[f"val/{label}/precision"] = prec
        metrics[f"val/{label}/recall"]    = rec
        metrics[f"val/{label}/accuracy"]  = acc
        metrics[f"val/{label}/n_pos"]     = n_pos
        metrics[f"val/{label}/n_neg"]     = n_neg

        if not (auc_roc != auc_roc):   # not nan
            aucs_roc.append(auc_roc)
            aucs_pr.append(auc_pr)
        f1s.append(f1)

    # Macro averages
    metrics["val/macro_auc_roc"] = sum(aucs_roc) / max(len(aucs_roc), 1)
    metrics["val/macro_auc_pr"]  = sum(aucs_pr)  / max(len(aucs_pr),  1)
    metrics["val/macro_f1"]      = sum(f1s)       / max(len(f1s),       1)

    return metrics, probs, preds, y


def log_confusion_matrices(probs, preds, y, epoch: int) -> None:
    """Log W&B confusion matrix for each label."""
    for i, label in enumerate(COMORBIDITY_LABELS):
        cm = confusion_matrix(y[:, i], preds[:, i], labels=[0, 1])
        wandb.log({
            f"val/{label}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y[:, i].tolist(),
                preds=preds[:, i].tolist(),
                class_names=["negative", "positive"],
                title=label,
            )
        }, step=epoch)


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_comorbidity_classifier(cfg: ComorbidityConfig | None = None) -> str:
    cfg    = cfg or ComorbidityConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    logger, csv_path = setup_logging(cfg.log_dir, prefix="comorbidity")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"{cfg.experiment_name}_seed{cfg.seed}",
        group   = cfg.experiment_name,
        config  = cfg.to_dict(),
        notes   = cfg.wandb_notes or None,
        tags    = cfg.wandb_tags  or [],
        reinit  = True,
    )

    logger.info("=" * 60)
    logger.info("Comorbidity Probe — OmniRad → GatedAttn → Classifier")
    logger.info(f"W&B run  : {run.url}")
    logger.info(f"Device   : {device}")
    logger.info("=" * 60)

    # ── Data ───────────────────────────────────────────────────────────────
    _ds_kwargs = dict(
        rootdir        = cfg.root_dir,
        target_spacing = cfg.target_spacing,
        target_shape   = cfg.target_shape,
        split_file     = cfg.json_path,
        metadata_path  = os.path.join(cfg.root_dir, "kits23.json"),
    )
    # Use full volume (val mode) for both — we need complete anatomy for
    # comorbidity prediction, not a 16-slice tumour crop
    train_ds = KitsDataset(**_ds_kwargs, mode="train_sur")
    val_ds   = KitsDataset(**_ds_kwargs, mode="val")

    _dl = dict(
        batch_size         = 1,
        num_workers        = cfg.num_workers,
        pin_memory         = False,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_dl)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_dl)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Comorbidity labels ─────────────────────────────────────────────────
    metadata_path = os.path.join(cfg.root_dir, "kits23.json")
    logger.info("Loading comorbidity labels...")
    all_ids = train_ds.cases + val_ds.cases
    label_dict, pos_weight, prevalence = load_comorbidity_labels(metadata_path, all_ids)

    wandb.log({f"data/prevalence_{l}": v for l, v in prevalence.items()}, step=0)
    wandb.log({f"data/pos_weight_{l}": pos_weight[i].item()
               for i, l in enumerate(COMORBIDITY_LABELS)}, step=0)

    # ── Frozen OmniRad ─────────────────────────────────────────────────────
    omnirad = OmniRadEncoder(device=device, frozen=True)
    logger.info("OmniRad loaded and frozen.")

    # ── Trainable pooling ──────────────────────────────────────────────────
    pooling = GatedAttentionPooling(
        embed_dim   = cfg.embed_dim,
        hidden_size = cfg.attn_hidden_size,
        dropout     = cfg.attn_dropout,
    ).to(device)
    n_pool = sum(p.numel() for p in pooling.parameters())
    logger.info(f"GatedAttentionPooling  params: {n_pool:,}")

    # ── Trainable classifier ───────────────────────────────────────────────
    classifier = ComorbidityClassifier(
        embed_dim  = cfg.embed_dim,
        hidden_dim = cfg.classifier_hidden_dim,
        dropout    = cfg.classifier_dropout,
    ).to(device)
    n_cls = sum(p.numel() for p in classifier.parameters())
    logger.info(f"ComorbidityClassifier  params: {n_cls:,}")
    wandb.log({"model/pooling_params": n_pool, "model/classifier_params": n_cls}, step=0)

    # ── Loss ───────────────────────────────────────────────────────────────
    # Per-label weighted BCE — each label has its own pos_weight to handle
    # the severe class imbalance (some comorbidities have <5% prevalence)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight = pos_weight.to(device),
        reduction  = "none",   # keep per-label losses for logging
    )

    # ── Optimiser ──────────────────────────────────────────────────────────
    trainable_params = list(pooling.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(trainable_params, lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch      = 1
    best_macro_auc   = -1.0
    no_improve_ctr   = 0

    if cfg.resume_path and os.path.exists(cfg.resume_path):
        saved = torch.load(cfg.resume_path, map_location=device)
        pooling.load_state_dict(saved["pooling_state"])
        classifier.load_state_dict(saved["classifier_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch    = saved["epoch"] + 1
        best_macro_auc = saved.get("val_macro_auc", -1.0)
        no_improve_ctr = saved.get("no_improve_ctr", 0)
        logger.info(f"Resumed from epoch {saved['epoch']}")

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        header = ["epoch", "lr", "train_loss_total"]
        header += [f"train_loss_{l}" for l in COMORBIDITY_LABELS]
        header += ["val_macro_auc_roc", "val_macro_auc_pr", "val_macro_f1"]
        header += [f"val_{l}_auc_roc" for l in COMORBIDITY_LABELS]
        csv.writer(f).writerow(header)

    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'='*60}\nEpoch {epoch}/{cfg.num_epochs}  LR: {lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────
        pooling.train()
        classifier.train()

        tr_loss_total = 0.0
        tr_loss_label = torch.zeros(N_LABELS)
        tr_grad_pool  = tr_grad_cls = 0.0
        tr_n = 0

        pbar = tqdm(train_loader, desc=f"  Train [{epoch}]", leave=False)
        for batch in pbar:
            ct      = batch["ct"][0]      # (D, H, W)
            gt_mask = batch["mask"][0]    # (D, H, W)
            cid     = batch["caseid"][0]

            if cid not in label_dict:
                continue
            targets = label_dict[cid].to(device)   # (5,)

            # Step 1: frozen OmniRad (no grad)
            slice_embs = get_slice_embeddings(ct, gt_mask, omnirad, cfg)

            # Step 2: trainable pooling (grad active, stays on GPU)
            emb, attn_w = pool_embeddings(slice_embs, pooling, device)

            # Step 3: classify
            logits = classifier(emb)               # (5,)

            # Per-label BCE loss then mean
            loss_per_label = criterion(logits, targets)   # (5,)
            loss = loss_per_label.mean()

            optimizer.zero_grad()
            loss.backward()

            # Gradient norms before clipping
            def _gnorm(ps):
                g = [p.grad.detach().norm() for p in ps if p.grad is not None]
                return torch.stack(g).norm().item() if g else 0.0

            tr_grad_pool += _gnorm(pooling.parameters())
            tr_grad_cls  += _gnorm(classifier.parameters())

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            tr_loss_total += loss.item()
            tr_loss_label += loss_per_label.detach().cpu()
            tr_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = max(tr_n, 1)
        avg_tr_loss       = tr_loss_total / n
        avg_tr_loss_label = tr_loss_label / n
        avg_grad_pool     = tr_grad_pool  / n
        avg_grad_cls      = tr_grad_cls   / n

        # ── Val ────────────────────────────────────────────────────────────
        pooling.eval()
        classifier.eval()

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attn_max: list[float]      = []
        all_attn_ent: list[float]      = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   [{epoch}]", leave=False):
                ct      = batch["ct"][0]
                gt_mask = batch["mask"][0]
                cid     = batch["caseid"][0]

                if cid not in label_dict:
                    continue

                slice_embs = get_slice_embeddings(ct, gt_mask, omnirad, cfg)
                emb, attn_w = pool_embeddings(slice_embs, pooling, device)
                logits = classifier(emb).cpu()

                all_logits.append(logits)
                all_labels.append(label_dict[cid])

                # Attention diagnostics
                all_attn_max.append(attn_w.max().item())
                ent = -(attn_w * (attn_w + 1e-8).log()).sum().item()
                all_attn_ent.append(ent)

        scheduler.step()

        if not all_logits:
            logger.warning("No val patients had comorbidity labels — skipping val metrics")
            continue

        logits_tensor = torch.stack(all_logits)   # (N, 5)
        labels_tensor = torch.stack(all_labels)   # (N, 5)

        val_metrics, probs, preds, y = compute_label_metrics(logits_tensor, labels_tensor)
        macro_auc = val_metrics["val/macro_auc_roc"]

        # ── Logging ────────────────────────────────────────────────────────
        logger.info(f"  Train → total_loss: {avg_tr_loss:.4f}")
        for i, lbl in enumerate(COMORBIDITY_LABELS):
            short = lbl.replace("_", " ")
            logger.info(
                f"    {short:<45s}  "
                f"train_loss={avg_tr_loss_label[i]:.4f}  "
                f"auc_roc={val_metrics[f'val/{lbl}/auc_roc']:.4f}  "
                f"f1={val_metrics[f'val/{lbl}/f1']:.4f}  "
                f"recall={val_metrics[f'val/{lbl}/recall']:.4f}"
            )
        logger.info(
            f"  Val  → macro_auc_roc={macro_auc:.4f}  "
            f"macro_f1={val_metrics['val/macro_f1']:.4f}  "
            f"(best: {best_macro_auc:.4f})"
        )

        wb_log = {
            "train/lr":           lr,
            "train/loss_total":   avg_tr_loss,
            "train/grad_pooling": avg_grad_pool,
            "train/grad_classif": avg_grad_cls,
            "val/attn_max":       sum(all_attn_max) / max(len(all_attn_max), 1),
            "val/attn_entropy":   sum(all_attn_ent) / max(len(all_attn_ent), 1),
            **val_metrics,
        }
        for i, lbl in enumerate(COMORBIDITY_LABELS):
            wb_log[f"train/loss_{lbl}"] = avg_tr_loss_label[i].item()

        wandb.log(wb_log, step=epoch)

        with open(csv_path, "a", newline="") as f:
            row = [epoch, f"{lr:.2e}", f"{avg_tr_loss:.6f}"]
            row += [f"{avg_tr_loss_label[i]:.6f}" for i in range(N_LABELS)]
            row += [
                f"{val_metrics['val/macro_auc_roc']:.4f}",
                f"{val_metrics['val/macro_auc_pr']:.4f}",
                f"{val_metrics['val/macro_f1']:.4f}",
            ]
            row += [f"{val_metrics[f'val/{l}/auc_roc']:.4f}" for l in COMORBIDITY_LABELS]
            csv.writer(f).writerow(row)

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt = {
            "epoch":            epoch,
            "pooling_state":    pooling.state_dict(),
            "classifier_state": classifier.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "val_macro_auc":    macro_auc,
            "no_improve_ctr":   no_improve_ctr,
            "config":           cfg.to_dict(),
        }
        torch.save(ckpt, cfg.last_ckpt)

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            no_improve_ctr = 0
            torch.save(ckpt, cfg.best_ckpt)
            logger.info(f"  ✓ Best saved (macro_auc_roc={best_macro_auc:.4f})")
            # Log confusion matrices at each new best
            log_confusion_matrices(probs, preds, y, epoch)
            wandb.run.summary.update({
                "best_macro_auc_roc": best_macro_auc,
                "best_epoch":         epoch,
                **{f"best_{k}": v for k, v in val_metrics.items()
                   if "auc_roc" in k or "f1" in k},
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

    logger.info(f"\nDone. Best val macro AUC-ROC: {best_macro_auc:.4f}")
    logger.info(
        "Interpretation guide:\n"
        "  AUC-ROC > 0.7  → OmniRad features carry signal for this comorbidity\n"
        "  AUC-ROC ~ 0.5  → features are uninformative (random)\n"
        "  High recall, low precision → model predicts positive too often\n"
        "  Low recall → model misses most positive cases (check pos_weight)"
    )
    wandb.finish()
    return cfg.best_ckpt


if __name__ == "__main__":
    train_comorbidity_classifier()