"""Reproducibility helpers: global seeding and lightweight experiment logging."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from .config import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed all RNGs (Python, NumPy, and PyTorch if available) for reproducibility.

    Also enables deterministic cuDNN behaviour so that GPU runs are repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    """Record per-epoch metrics and run configuration to JSON for reproducibility."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[int, dict[str, float]] = {}

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record metrics for one epoch and flush them to metrics.json."""
        self.metrics[int(epoch)] = {k: float(v) for k, v in metrics.items()}
        with open(self.experiment_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)

    def log_config(self, config: dict[str, Any]) -> None:
        """Persist the run configuration to config.json."""
        with open(self.experiment_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, default=str)
