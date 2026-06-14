"""Neural collaborative filtering with PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, RATING_SCALE
from .metrics import clip_predictions
from .preprocessing import global_mean

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class _NCFNetwork(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Run the forward pass: embed user/item indices and return predicted ratings."""
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.mlp(x).squeeze(1)


class NeuralCFRecommender:
    """Rating-prediction Neural CF model with user and movie embeddings."""

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 8,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        random_state: int = RANDOM_STATE,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for NeuralCFRecommender.")
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_: _NCFNetwork | None = None
        self.user_idx_: dict[int, int] = {}
        self.item_idx_: dict[int, int] = {}
        self.global_mean_: float = 0.0
        self.history_: list[float] = []

    def fit(self, train: pd.DataFrame) -> NeuralCFRecommender:
        """Train the NeuralCF network on the training ratings and return self."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.global_mean_ = global_mean(train)

        user_ids = sorted(train["user_id"].unique())
        item_ids = sorted(train["movie_id"].unique())
        self.user_idx_ = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
        self.item_idx_ = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

        user_tensor = torch.tensor(
            [self.user_idx_[int(user_id)] for user_id in train["user_id"]],
            dtype=torch.long,
        )
        item_tensor = torch.tensor(
            [self.item_idx_[int(item_id)] for item_id in train["movie_id"]],
            dtype=torch.long,
        )
        rating_tensor = torch.tensor(
            train["rating"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.model_ = _NCFNetwork(
            n_users=len(self.user_idx_),
            n_items=len(self.item_idx_),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        self.history_ = []

        self.model_.train()
        for _ in range(self.epochs):
            epoch_losses = []
            for users, items, ratings in loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                preds = self.model_(users, items)
                loss = loss_fn(preds, ratings)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.history_.append(float(np.mean(epoch_losses)))
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for one (user_id, movie_id) pair."""
        frame = pd.DataFrame({"user_id": [user_id], "movie_id": [movie_id]})
        return float(self.predict_batch(frame)[0])

    def predict_for_user(self, user_id: int, movie_ids: list[int]) -> pd.Series:
        """Vectorized prediction for one user over many candidate movies.

        Returns a Series indexed by movie_id, matching the interface used by the
        other recommenders so the Streamlit recommendation pipeline can score all
        candidates in a single pass.
        """
        frame = pd.DataFrame({"user_id": user_id, "movie_id": list(movie_ids)})
        preds = self.predict_batch(frame)
        return pd.Series(preds, index=list(movie_ids), dtype=float)

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_batch().")

        preds = np.full(len(test), self.global_mean_, dtype=float)
        known_positions = []
        user_indices = []
        item_indices = []

        for pos, row in enumerate(test.itertuples(index=False)):
            user_id = int(row.user_id)
            item_id = int(row.movie_id)
            if user_id in self.user_idx_ and item_id in self.item_idx_:
                known_positions.append(pos)
                user_indices.append(self.user_idx_[user_id])
                item_indices.append(self.item_idx_[item_id])

        if not known_positions:
            return clip_predictions(preds, *RATING_SCALE)

        self.model_.eval()
        with torch.no_grad():
            users = torch.tensor(user_indices, dtype=torch.long, device=self.device)
            items = torch.tensor(item_indices, dtype=torch.long, device=self.device)
            model_preds = self.model_(users, items).detach().cpu().numpy()
        preds[np.array(known_positions)] = model_preds
        return clip_predictions(preds, *RATING_SCALE)
