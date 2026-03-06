"""
utils/wandb_utils.py
====================
Helpers for accumulating / averaging / logging the debug statistics
emitted by CEDiceLoss, and for flattening them into W&B-compatible dicts.
"""


def empty_debug_acc() -> dict:
    return {
        "max_bg_prob":    0.0,
        "max_fg_prob":    0.0,
        "intersect":      None,
        "cardinality":    None,
        "dice_per_class": None,
    }


def accumulate_debug_stats(acc: dict, stats: dict) -> None:
    """In-place accumulation across batches."""
    acc["max_bg_prob"] = max(acc["max_bg_prob"], stats["max_bg_prob"])
    acc["max_fg_prob"] = max(acc["max_fg_prob"], stats["max_fg_prob"])

    for key in ("intersect", "cardinality", "dice_per_class"):
        if acc[key] is None:
            acc[key] = list(stats[key])
        else:
            acc[key] = [a + b for a, b in zip(acc[key], stats[key])]


def average_debug_stats(acc: dict, n: int) -> dict:
    """Divide list-based accumulated stats by n."""
    return {
        "max_bg_prob":    acc["max_bg_prob"],
        "max_fg_prob":    acc["max_fg_prob"],
        "intersect":      [v / n for v in acc["intersect"]],
        "cardinality":    [v / n for v in acc["cardinality"]],
        "dice_per_class": [v / n for v in acc["dice_per_class"]],
    }


def log_debug_stats(logger, prefix: str, stats: dict) -> None:
    """Write a human-readable summary to the logger."""
    logger.info(
        f"  {prefix} Debug → "
        f"max_bg_prob: {stats['max_bg_prob']:.4f}  |  "
        f"max_fg_prob: {stats['max_fg_prob']:.4f}"
    )
    if stats["dice_per_class"]:
        per = "  ".join(f"c{i}: {v:.4f}" for i, v in enumerate(stats["dice_per_class"]))
        logger.info(f"  {prefix} Debug → soft_dice_per_class: {per}")
    if stats["intersect"]:
        logger.info(
            f"  {prefix} Debug → avg intersect  : " +
            "  ".join(f"c{i}: {v:.2f}" for i, v in enumerate(stats["intersect"]))
        )
        logger.info(
            f"  {prefix} Debug → avg cardinality: " +
            "  ".join(f"c{i}: {v:.2f}" for i, v in enumerate(stats["cardinality"]))
        )


def debug_stats_to_wandb(prefix: str, stats: dict) -> dict:
    """Flatten debug stats into a flat dict ready for wandb.log()."""
    d: dict = {
        f"{prefix}/debug/max_bg_prob": stats["max_bg_prob"],
        f"{prefix}/debug/max_fg_prob": stats["max_fg_prob"],
    }
    for key, tag in (
        ("dice_per_class", "soft_dice_class"),
        ("intersect",      "intersect_class"),
        ("cardinality",    "cardinality_class"),
    ):
        if stats[key]:
            for i, v in enumerate(stats[key]):
                d[f"{prefix}/debug/{tag}_{i}"] = v
    return d