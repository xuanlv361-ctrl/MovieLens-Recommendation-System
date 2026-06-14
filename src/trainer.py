"""Training-loop utilities for the PyTorch NeuralCF model.

Provides a small Trainer class (with logging, optional tqdm progress bars, per-epoch
loss history, and optional best-model checkpointing) so the deep-learning training
flow follows the project code standard instead of an inline loop.
"""

from __future__ import annotations

import logging

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional dependency
    np = None
    torch = None
    DataLoader = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)


class NCFTrainer:
    """Drive NeuralCF training with logging, optional progress bars and loss history."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        val_loader: DataLoader | None = None,
        show_progress: bool = False,
        checkpoint_path: str | None = None,
        experiment_logger=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.show_progress = show_progress and tqdm is not None
        self.checkpoint_path = checkpoint_path
        self.experiment_logger = experiment_logger
        self.history: list[float] = []

    def train_epoch(self) -> float:
        """Run one training epoch and return its mean batch loss."""
        self.model.train()
        epoch_losses = []
        iterable = self.train_loader
        if self.show_progress:
            iterable = tqdm(self.train_loader, desc="Training", leave=False)
        for users, items, ratings in iterable:
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(users, items)
            loss = self.criterion(preds, ratings)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        return float(np.mean(epoch_losses))

    def validate(self) -> float:
        """Return the mean validation loss over val_loader (NaN if no val set)."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for users, items, ratings in self.val_loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                preds = self.model(users, items)
                val_losses.append(float(self.criterion(preds, ratings).detach().cpu()))
        return float(np.mean(val_losses)) if val_losses else float("nan")

    def train(self) -> list[float]:
        """Run the full training loop and return the per-epoch training-loss history."""
        self.history = []
        best_loss = float("inf")
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch()
            self.history.append(epoch_loss)

            metrics = {"train_loss": epoch_loss}
            if self.val_loader is not None:
                val_loss = self.validate()
                metrics["val_loss"] = val_loss
                logger.info(
                    "Epoch %03d/%03d - train_loss: %.6f, val_loss: %.6f",
                    epoch + 1, self.epochs, epoch_loss, val_loss,
                )
            else:
                logger.info(
                    "Epoch %03d/%03d - train_loss: %.6f", epoch + 1, self.epochs, epoch_loss
                )

            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(epoch + 1, metrics)

            monitor = metrics.get("val_loss", epoch_loss)
            if self.checkpoint_path is not None and monitor < best_loss:
                best_loss = monitor
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info("Saved best model (loss %.6f) to %s", monitor, self.checkpoint_path)
        return self.history
