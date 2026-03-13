"""
utils/logging_utils.py
=======================
Shared logging setup used by both training scripts.
"""

import logging
import os
from dataclasses import asdict
from datetime import datetime


def setup_logging(log_dir: str, prefix: str = "train") -> tuple[logging.Logger, str]:
    """
    Create a timed log file + CSV file, attach file + stream handlers.

    Returns
    -------
    logger   : configured Logger instance
    csv_path : path to the metrics CSV file
    """
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{ts}.log")
    csv_path = os.path.join(log_dir, f"{prefix}_metrics_{ts}.csv")

    # Reset handlers so re-importing during the launcher doesn't duplicate output
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__), csv_path


# ── Section ordering for the config dump ──────────────────────────────────────
# Keys are printed in this order; anything not listed here is appended at the
# end under "other".  Using explicit groups makes the log easy to scan.

_UNET_SECTIONS: dict[str, list[str]] = {
    "Experiment":   ["experiment_name", "seed"],
    "Paths":        ["root_dir", "json_path", "run_dir",
                     "best_ckpt", "last_ckpt", "resume_path"],
    "Data":         ["target_spacing", "target_shape", "tumour_crop_p"],
    "Model":        ["num_classes", "base_channels", "trilinear"],
    "Training":     ["num_epochs", "batch_size", "learning_rate",
                     "weight_decay", "accumulation_steps",
                     "early_stop_patience", "num_workers"],
    "Loss":         ["ce_weight", "class_weights"],
    "Sliding win":  ["sw_window", "sw_stride"],
    "W&B":          ["wandb_project"],
}

_SURVIVAL_SECTIONS: dict[str, list[str]] = {
    "Experiment":   ["experiment_name", "seed"],
    "Paths":        ["root_dir", "json_path", "unet_ckpt", "run_dir",
                     "best_ckpt", "last_ckpt", "resume_path"],
    "Data":         ["target_spacing", "target_shape"],
    "UNet (frozen)":["num_classes", "unet_base_channels", "unet_trilinear",
                     "sw_window", "sw_stride"],
    "Mask source":  ["use_gt_mask"],
    "OmniRad":      ["embed_dim", "omni_batch"],
    "Slice pool":   ["slice_pooling", "attn_hidden_size", "attn_dropout"],
    "EGMDM head":   ["egmdm_E", "egmdm_K", "egmdm_hidden_size", "egmdm_dropout"],
    "Loss":         ["lambda_div", "lambda_ent"],
    "Training":     ["num_epochs", "learning_rate", "weight_decay",
                     "early_stop_patience", "num_workers"],
    "W&B":          ["wandb_project"],
}

_SURVIVAL_RGB_SECTIONS: dict[str, list[str]] = {
    "Experiment":   ["experiment_name", "seed"],
    "Paths":        ["root_dir", "json_path", "run_dir",
                     "best_ckpt", "last_ckpt", "resume_path"],
    "Data":         ["target_spacing", "target_shape"],
    "HU windows":   ["hu_windows"],
    "OmniRad":      ["embed_dim", "omni_batch"],
    "Slice pool":   ["slice_pooling", "attn_hidden_size", "attn_dropout"],
    "EGMDM head":   ["egmdm_E", "egmdm_K", "egmdm_hidden_size", "egmdm_dropout"],
    "Loss":         ["lambda_div", "lambda_ent"],
    "Training":     ["num_epochs", "learning_rate", "weight_decay",
                     "early_stop_patience", "num_workers"],
    "W&B":          ["wandb_project"],
    "Device":       ["device"],
}


def log_config(logger: logging.Logger, cfg) -> None:
    """
    Print every config field to the logger in a clearly sectioned block.

    Works with both UNetConfig and SurvivalConfig.  Falls back to a flat
    alphabetical dump for any unknown config type.

    Also calls wandb.config.update() so that any fields not captured by
    wandb.init(config=...) are still recorded — this is a no-op if W&B
    is in offline mode.
    """
    import wandb

    cfg_dict = asdict(cfg)

    # Pick the right section map
    class_name = type(cfg).__name__
    if "UNet" in class_name:
        sections = _UNET_SECTIONS
    elif "RGB" in class_name:
        sections = _SURVIVAL_RGB_SECTIONS
    elif "Survival" in class_name:
        sections = _SURVIVAL_SECTIONS
    else:
        sections = {"Config": list(cfg_dict.keys())}

    logged_keys: set[str] = set()

    logger.info("")
    logger.info("┌─ Full Config " + "─" * 46)

    for section, keys in sections.items():
        # Filter to keys that actually exist in this config
        present = [k for k in keys if k in cfg_dict]
        if not present:
            continue
        logger.info(f"│  [{section}]")
        for k in present:
            v = cfg_dict[k]
            logger.info(f"│    {k:<28} = {v}")
            logged_keys.add(k)

    # Catch any fields not covered by the section map (future-proofing)
    remainder = {k: v for k, v in cfg_dict.items() if k not in logged_keys}
    if remainder:
        logger.info("│  [other]")
        for k, v in sorted(remainder.items()):
            logger.info(f"│    {k:<28} = {v}")

    logger.info("└" + "─" * 59)
    logger.info("")

    # Ensure W&B has every field, including any added after wandb.init()
    try:
        wandb.config.update(cfg_dict, allow_val_change=True)
    except Exception:
        pass  # W&B not initialised yet or offline — silently skip