"""User-based collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K_NEIGHBORS, MIN_COMMON_RATINGS, RATING_SCALE, SIMILARITY_METRIC
from .metrics import clip_predictions
from .preprocessing import build_user_item_matrix, global_mean, mean_center_by_user
from .similarity import pairwise_similarity_matrix


class UserBasedCF:
    """Predict ratings from similar users' centered ratings."""

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
        self._user_means: pd.Series | None = None
        self._sim: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    def fit(self, train: pd.DataFrame) -> UserBasedCF:
        self._global_mean = global_mean(train)
        self._matrix = build_user_item_matrix(train, fill_value=np.nan)
        self._centered, self._user_means = mean_center_by_user(self._matrix)
        self._sim = pairwise_similarity_matrix(
            self._centered,
            metric=self.metric,
            min_common=self.min_common,
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if self._matrix is None or self._sim is None or self._user_means is None:
            raise RuntimeError("Call fit() before predict().")

        if user_id not in self._matrix.index or movie_id not in self._matrix.columns:
            return self._global_mean

        if user_id not in self._sim.index:
            return self._global_mean

        # Users who rated this movie
        col = self._matrix[movie_id]
        raters = col.dropna().index
        if len(raters) == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        sims = self._sim.loc[user_id, raters].drop(labels=[user_id], errors="ignore")
        sims = sims[sims > 0].sort_values(ascending=False).head(self.k)

        if len(sims) == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        num = 0.0
        den = 0.0
        for other_id, sim in sims.items():
            raw = self._matrix.loc[other_id, movie_id]
            if np.isnan(raw):
                continue
            centered = raw - self._user_means[other_id]
            num += sim * centered
            den += abs(sim)

        if den == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        user_mean = float(self._user_means.get(user_id, self._global_mean))
        pred = user_mean + num / den
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        preds = [
            self.predict(int(row.user_id), int(row.movie_id))
            for row in test.itertuples(index=False)
        ]
        return np.array(preds, dtype=float)
