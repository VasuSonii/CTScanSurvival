"""
run_experiments.py — Multi-seed / hyperparameter experiment launcher
=====================================================================

Usage
-----
Run everything defined in UNET_GRID and SURVIVAL_GRID:
    python -m kits21.run_experiments

Run only Phase 1:
    python -m kits21.run_experiments --phase 1

Run only Phase 2 (Phase 1 checkpoints must already exist):
    python -m kits21.run_experiments --phase 2

How it works
------------
1. UNET_GRID defines a list of hyperparameter overrides + seeds.
   For every (override, seed) pair, dataclasses.replace() creates an
   isolated UNetConfig that writes to its own run_dir.

2. After each Phase 1 run, the returned best.pth path is stored so
   Phase 2 can point unet_ckpt at the correct seed-matched checkpoint.

3. SURVIVAL_GRID follows the same pattern, pairing each survival
   experiment with its corresponding Phase 1 result.

Adding a new experiment
-----------------------
Add an entry to UNET_EXPERIMENTS or SURVIVAL_EXPERIMENTS below.
Each entry is a dict of field overrides accepted by the config dataclass.
Seeds are applied on top of each experiment dict automatically.
"""

import argparse
import dataclasses
import logging
import sys
import traceback
from typing import Any
import os
os.environ["WANDB_MODE"] = "offline"
from configs.unet_config     import UNetConfig
from configs.survival_config import SurvivalConfig
from train_unet              import train_unet
from train_survival          import train_survival

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# ── Define your experiments here ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 2024]

# Each entry becomes a separate UNetConfig with the listed fields overridden.
# The 'experiment_name' key is required in every entry.
UNET_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_name": "unet_baseline",
        "learning_rate":   1e-4,
        "ce_weight":       0.1,
        "base_channels":   32,
    },
    {
        "experiment_name": "unet_high_lr",
        "learning_rate":   3e-4,
        "ce_weight":       0.1,
        "base_channels":   32,
    },
    {
        "experiment_name": "unet_ce_balanced",
        "learning_rate":   1e-4,
        "ce_weight":       0.3,
        "base_channels":   32,
    },
    {
        "experiment_name": "unet_large",
        "learning_rate":   1e-4,
        "ce_weight":       0.1,
        "base_channels":   64,
    },
]

# Each entry becomes a SurvivalConfig.
# 'unet_experiment_name' tells the launcher which Phase 1 experiment to
# pull the frozen UNet from.  It must match an entry in UNET_EXPERIMENTS.
SURVIVAL_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_name":      "survival_baseline",
        "unet_experiment_name": "unet_baseline",   # ← links to Phase 1
        "egmdm_E":              3,
        "egmdm_K":              10,
        "learning_rate":        1e-4,
    },
    {
        "experiment_name":      "survival_large_mixture",
        "unet_experiment_name": "unet_baseline",
        "egmdm_E":              5,
        "egmdm_K":              15,
        "learning_rate":        1e-4,
    },
    {
        "experiment_name":      "survival_high_lr",
        "unet_experiment_name": "unet_baseline",
        "egmdm_E":              3,
        "egmdm_K":              10,
        "learning_rate":        3e-4,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _make_unet_cfg(overrides: dict[str, Any], seed: int) -> UNetConfig:
    """Create an isolated UNetConfig for one (experiment, seed) pair."""
    base = UNetConfig(seed=seed)
    # Pop non-dataclass keys before passing to replace()
    clean = {k: v for k, v in overrides.items() if k != "unet_experiment_name"}
    return dataclasses.replace(base, seed=seed, **clean)


def _make_survival_cfg(
    overrides:       dict[str, Any],
    seed:            int,
    unet_best_ckpt:  str,
) -> SurvivalConfig:
    """Create an isolated SurvivalConfig for one (experiment, seed) pair."""
    base  = SurvivalConfig(seed=seed)
    clean = {k: v for k, v in overrides.items() if k != "unet_experiment_name"}
    return dataclasses.replace(base, seed=seed, unet_ckpt=unet_best_ckpt, **clean)


def _run_safe(fn, cfg, phase_label: str) -> str | None:
    """Run a training function; catch and log exceptions without aborting the sweep."""
    tag = f"{phase_label} | {cfg.experiment_name} | seed={cfg.seed}"
    logger.info(f"\n{'#'*70}\n  Starting  {tag}\n{'#'*70}")
    try:
        result = fn(cfg)
        logger.info(f"  Finished  {tag}  →  {result}")
        return result
    except Exception:
        logger.error(f"  FAILED    {tag}\n{traceback.format_exc()}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ── Runner ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1() -> dict[tuple[str, int], str]:
    """
    Run all Phase 1 experiments × seeds.

    Returns
    -------
    completed : {(experiment_name, seed): best_ckpt_path}
    """
    completed: dict[tuple[str, int], str] = {}

    for exp in UNET_EXPERIMENTS:
        for seed in SEEDS:
            cfg  = _make_unet_cfg(exp, seed)
            path = _run_safe(train_unet, cfg, "Phase 1")
            if path:
                completed[(exp["experiment_name"], seed)] = path

    return completed


def run_phase2(phase1_results: dict[tuple[str, int], str]) -> None:
    """Run all Phase 2 experiments × seeds, wiring in the correct UNet checkpoint."""
    for exp in SURVIVAL_EXPERIMENTS:
        unet_exp_name = exp.get("unet_experiment_name", "unet_baseline")
        for seed in SEEDS:
            unet_ckpt = phase1_results.get((unet_exp_name, seed))
            if unet_ckpt is None:
                logger.warning(
                    f"Skipping survival '{exp['experiment_name']}' seed={seed}: "
                    f"no Phase 1 result for ('{unet_exp_name}', seed={seed})."
                )
                continue
            cfg = _make_survival_cfg(exp, seed, unet_ckpt)
            _run_safe(train_survival, cfg, "Phase 2")


# ══════════════════════════════════════════════════════════════════════════════
# ── Entry point ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="KiTS21 experiment launcher")
    parser.add_argument(
        "--phase",
        type    = int,
        choices = [1, 2],
        default = None,
        help    = "Run only phase 1 or phase 2. Default: run both sequentially.",
    )
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1()

    elif args.phase == 2:
        # Phase 1 results must already exist on disk; reconstruct paths.
        phase1_results: dict[tuple[str, int], str] = {}
        import os
        for exp in UNET_EXPERIMENTS:
            for seed in SEEDS:
                cfg  = _make_unet_cfg(exp, seed)
                path = cfg.best_ckpt
                if os.path.exists(path):
                    phase1_results[(exp["experiment_name"], seed)] = path
                else:
                    logger.warning(
                        f"Phase 1 checkpoint not found for "
                        f"('{exp['experiment_name']}', seed={seed}): {path}"
                    )
        run_phase2(phase1_results)

    else:
        # Default: run both phases end-to-end
        phase1_results = run_phase1()
        run_phase2(phase1_results)

    logger.info("\nAll experiments finished.")


if __name__ == "__main__":
    main()