"""Matrix factorization with sklearn TruncatedSVD (no scikit-surprise)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .config import RANDOM_STATE, RATING_SCALE, SVD_N_COMPONENTS, SVD_N_ITER
from .metrics import clip_predictions
from .preprocessing import global_mean


class SVDRecommender:
    """
    Latent-factor model: R ≈ U @ V^T on mean-centered ratings.

    Missing entries are omitted from the sparse matrix (stored as implicit zeros
    in CSR). Predictions add back per-user means and clip to the rating scale.
    """

    def __init__(
        self,
        n_components: int = SVD_N_COMPONENTS,
        n_iter: int = SVD_N_ITER,
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self._svd: TruncatedSVD | None = None
        self._user_factors: np.ndarray | None = None
        self._user_idx: dict[int, int] = {}
        self._movie_idx: dict[int, int] = {}
        self._user_means: pd.Series | None = None
        self._global_mean: float = 0.0

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        self._global_mean = global_mean(train)
        self._user_means = train.groupby("user_id")["rating"].mean()

        user_ids = sorted(train["user_id"].unique())
        movie_ids = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(user_ids)}
        self._movie_idx = {m: i for i, m in enumerate(movie_ids)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row in train.itertuples(index=False):
            u = int(row.user_id)
            m = int(row.movie_id)
            centered = float(row.rating) - float(self._user_means[u])
            rows.append(self._user_idx[u])
            cols.append(self._movie_idx[m])
            data.append(centered)

        n_users = len(user_ids)
        n_movies = len(movie_ids)
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_movies),
            dtype=np.float64,
        )

        n_comp = min(self.n_components, min(matrix.shape) - 1)
        if n_comp < 1:
            raise ValueError("Not enough ratings to fit TruncatedSVD.")

        self._svd = TruncatedSVD(
            n_components=n_comp,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._user_factors = self._svd.fit_transform(matrix)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if self._svd is None or self._user_factors is None or self._user_means is None:
            raise RuntimeError("Call fit() before predict().")

        if user_id not in self._user_idx or movie_id not in self._movie_idx:
            return self._global_mean

        u = self._user_idx[user_id]
        m = self._movie_idx[movie_id]
        latent = float(self._user_factors[u] @ self._svd.components_[:, m])
        pred = float(self._user_means[user_id]) + latent
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        if self._svd is None or self._user_factors is None or self._user_means is None:
            raise RuntimeError("Call fit() before predict_batch().")

        preds = np.empty(len(test), dtype=float)
        components = self._svd.components_

        for i, row in enumerate(test.itertuples(index=False)):
            user_id = int(row.user_id)
            movie_id = int(row.movie_id)
            if user_id not in self._user_idx or movie_id not in self._movie_idx:
                preds[i] = self._global_mean
                continue
            u = self._user_idx[user_id]
            m = self._movie_idx[movie_id]
            latent = float(self._user_factors[u] @ components[:, m])
            preds[i] = float(self._user_means[user_id]) + latent

        return clip_predictions(preds, *RATING_SCALE)
