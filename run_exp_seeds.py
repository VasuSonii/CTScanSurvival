"""
run_survival_seeds.py — Multi-GPU parallel Phase 2 survival training
=====================================================================
Usage:
    python run_survival_seeds.py

Strategy
--------
3 GPUs (cuda:0, cuda:1, cuda:2) → 3 workers running at a time.
Each worker is pinned to one GPU exclusively.
As soon as a seed finishes on a GPU, the next pending seed is
dispatched to that same GPU — no GPU ever sits idle while work remains.

Seeds:  8 total  →  wave 1: seeds 0-2 (one per GPU)
                 →  wave 2: seeds 3-5 (as wave 1 finishes)
                 →  wave 3: seeds 6-7 (as wave 2 finishes)

W&B is forced offline.
Results land in:
    runs/<experiment_name>/seed_<seed>/
"""

import logging
import os
import queue
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

os.environ["WANDB_MODE"] = "offline"

# ══════════════════════════════════════════════════════════════════════════════
# ── Configure here ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SEEDS = [456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

GPUS = ["cuda:1", "cuda:2", "cuda:3"]   # one worker slot per GPU

BASE_CONFIG = dict(
    experiment_name   = "survival_baseline",
    unet_ckpt         = "runs/unet_baseline/seed_42/best.pth",

    use_gt_mask       = True,

    slice_pooling     = "attention",
    attn_hidden_size  = 128,
    attn_dropout      = 0.25,

    egmdm_E           = 3,
    egmdm_K           = 10,
    egmdm_hidden_size = 256,
    egmdm_dropout     = 0.1,

    num_epochs            = 100,
    learning_rate         = 1e-4,
    weight_decay          = 1e-4,
    early_stop_patience   = 20,
    num_workers           = 16,   # 3 processes × 8 = 24 CPU workers total
)


# ══════════════════════════════════════════════════════════════════════════════
# ── Worker ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _worker(seed: int, device: str) -> tuple:
    """
    Runs in a spawned subprocess pinned to `device`.
    Returns (seed, device, best_ckpt_or_None, error_str).
    """
    os.environ["WANDB_MODE"] = "offline"

    import dataclasses
    from configs.survival_config import SurvivalConfig
    from train_survival          import train_survival

    try:
        cfg = dataclasses.replace(
            SurvivalConfig(seed=seed),
            **BASE_CONFIG,
            seed   = seed,
            device = device,
        )
        best_ckpt = train_survival(cfg)
        return seed, device, best_ckpt, ""
    except Exception:
        return seed, device, None, traceback.format_exc()


# ══════════════════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Print GPU inventory
    for gpu in GPUS:
        mem = torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        logger.info(f"  {gpu}  {mem:.0f} GB")
    logger.info(f"Seeds ({len(SEEDS)}): {SEEDS}")
    logger.info(f"Workers: {len(GPUS)} (one per GPU)\n")

    # gpu_pool is a queue of free GPU slots.
    # Each worker claims a GPU on start and returns it on finish.
    gpu_pool: queue.Queue = queue.Queue()
    for gpu in GPUS:
        gpu_pool.put(gpu)

    start_time = time.time()
    results: dict = {}
    errors:  dict = {}

    mp_ctx = torch.multiprocessing.get_context("spawn")

    # max_workers = number of GPUs — enforces one process per GPU
    with ProcessPoolExecutor(max_workers=len(GPUS), mp_context=mp_ctx) as pool:
        pending_seeds = list(SEEDS)
        futures: dict = {}   # future → (seed, device)

        # Seed the pool: submit one job per GPU to fill all slots immediately
        while pending_seeds and not gpu_pool.empty():
            seed   = pending_seeds.pop(0)
            device = gpu_pool.get()
            f      = pool.submit(_worker, seed, device)
            futures[f] = (seed, device)
            logger.info(f"  DISPATCH  seed={seed:<6}  →  {device}")

        # As each future completes, free its GPU and dispatch the next seed
        for future in as_completed(futures):
            seed_in, device_in = futures[future]

            try:
                seed_out, device_out, best_ckpt, err = future.result()
            except Exception:
                seed_out  = seed_in
                device_out = device_in
                best_ckpt  = None
                err        = traceback.format_exc()

            results[seed_out] = best_ckpt
            errors[seed_out]  = err

            if best_ckpt:
                logger.info(f"  DONE      seed={seed_out:<6}  {device_out}  →  {best_ckpt}")
            else:
                last = err.strip().splitlines()[-1] if err else "unknown"
                logger.error(f"  FAILED    seed={seed_out:<6}  {device_out}  {last}")

            # Return GPU to pool and immediately dispatch next pending seed
            gpu_pool.put(device_out)
            if pending_seeds:
                seed   = pending_seeds.pop(0)
                device = gpu_pool.get()
                f      = pool.submit(_worker, seed, device)
                futures[f] = (seed, device)
                logger.info(f"  DISPATCH  seed={seed:<6}  →  {device}")

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  SEED SWEEP SUMMARY")
    logger.info("=" * 70)

    passed = [(s, p) for s, p in results.items() if p is not None]
    failed = [(s, errors[s]) for s, p in results.items() if p is None]

    for seed, path in sorted(passed):
        logger.info(f"  seed={seed:<6}  OK      {path}")
    for seed, err in sorted(failed):
        last = err.strip().splitlines()[-1] if err else "unknown"
        logger.info(f"  seed={seed:<6}  FAILED  {last}")

    logger.info(
        f"\n  {len(passed)}/{len(SEEDS)} completed  "
        f"(wall time: {elapsed/60:.1f} min)"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()