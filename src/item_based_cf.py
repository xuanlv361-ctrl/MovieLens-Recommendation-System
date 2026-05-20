"""Item-based collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K_NEIGHBORS, MIN_COMMON_RATINGS, RATING_SCALE, SIMILARITY_METRIC
from .metrics import clip_predictions
from .preprocessing import build_user_item_matrix, global_mean, mean_center_by_item
from .similarity import pairwise_similarity_matrix


class ItemBasedCF:
    """Predict ratings from similar items' centered ratings."""

    def __init__(
        self,
        k: int = K_NEIGHBORS,
        min_common: int = MIN_COMMON_RATINGS,
        metric: str = SIMILARITY_METRIC,
    ):
        self.k = k
        self.min_common = min_common
        self.metric = metric
        self._matrix: pd.DataFrame | None = None
        self._centered: pd.DataFrame | None = None
        self._item_means: pd.Series | None = None
        self._sim: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    def fit(self, train: pd.DataFrame) -> "ItemBasedCF":
        self._global_mean = global_mean(train)
        self._matrix = build_user_item_matrix(train, fill_value=np.nan)
        self._centered, self._item_means = mean_center_by_item(self._matrix)
        centered_items = self._centered.T
        self._sim = pairwise_similarity_matrix(
            centered_items,
            metric=self.metric,
            min_common=self.min_common,
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if self._matrix is None or self._sim is None or self._item_means is None:
            raise RuntimeError("Call fit() before predict().")

        if user_id not in self._matrix.index or movie_id not in self._matrix.columns:
            return self._global_mean

        if movie_id not in self._sim.index:
            return self._global_mean

        user_row = self._matrix.loc[user_id]
        rated_items = user_row.dropna().index
        if len(rated_items) == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        sims = self._sim.loc[movie_id, rated_items].drop(labels=[movie_id], errors="ignore")
        sims = sims[sims > 0].sort_values(ascending=False).head(self.k)

        if len(sims) == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        num = 0.0
        den = 0.0
        for other_movie, sim in sims.items():
            raw = user_row[other_movie]
            if np.isnan(raw):
                continue
            centered = raw - self._item_means[other_movie]
            num += sim * centered
            den += abs(sim)

        if den == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        item_mean = float(self._item_means.get(movie_id, self._global_mean))
        pred = item_mean + num / den
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        preds = [
            self.predict(int(row.user_id), int(row.movie_id))
            for row in test.itertuples(index=False)
        ]
        return np.array(preds, dtype=float)

    def similar_items(self, movie_id: int, n: int = 10) -> pd.Series:
        """Return top-n most similar movies by item–item similarity (training fit required)."""
        if self._sim is None:
            raise RuntimeError("Call fit() before similar_items().")
        if movie_id not in self._sim.index:
            return pd.Series(dtype=float)
        sims = self._sim.loc[movie_id].drop(labels=[movie_id], errors="ignore")
        sims = sims[sims > 0].sort_values(ascending=False).head(n)
        return sims
