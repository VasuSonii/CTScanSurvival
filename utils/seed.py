"""
utils/seed.py
=============
Deterministic seeding for reproducible experiments.

Sets Python, NumPy, PyTorch (CPU + CUDA), and cuDNN flags so that
two runs with the same seed produce identical results, provided the
hardware and batch ordering are also identical.

Note: enabling deterministic CUDA ops (`torch.use_deterministic_algorithms`)
can reduce throughput.  It is opt-in here via `deterministic=True`.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all random-number generators used during training.

    Parameters
    ----------
    seed          : integer seed value
    deterministic : if True, force CUDA deterministic algorithms
                    (slower but fully reproducible on GPU)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # Required by some CUDA ops in deterministic mode
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Allow cuDNN auto-tuner for best throughput
        torch.backends.cudnn.benchmark = True