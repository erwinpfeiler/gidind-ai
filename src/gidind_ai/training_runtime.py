"""Deterministic runtime setup shared by cluster training jobs."""

from __future__ import annotations

import os
import random

import lightning as L
import numpy as np
import torch


def configure_training_seed(seed: int, *, deterministic: bool) -> None:
    """Seed every random source used by one training run.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA is initialised. The
    cluster launcher does that before importing the model code; setting it here
    also makes direct Python callers as reproducible as their import order
    permits.
    """
    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
