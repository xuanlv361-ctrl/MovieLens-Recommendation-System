"""Metadata-aware hybrid rating model."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RATING_SCALE
from .metrics import clip_predictions

NON_GENRE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
}


class MetadataHybridRegressor:
    """
    Hybrid recommender using user statistics, movie statistics, and movie metadata.

    The model is intentionally lightweight: it avoids deep-learning dependencies and
    gives an interpretable metadata baseline before moving to neural CF.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.global_mean_: float = 0.0
        self.user_features_: pd.DataFrame | None = None
        self.movie_features_: pd.DataFrame | None = None
        self.genre_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.pipeline_: Pipeline | None = None

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame) -> MetadataHybridRegressor:
        """Fit the model on the training ratings and return self."""
        self.global_mean_ = float(train["rating"].mean())
        self.genre_cols_ = [col for col in movies.columns if col not in NON_GENRE_COLUMNS]
        self.user_features_ = self._build_user_features(train)
        self.movie_features_ = self._build_movie_features(train, movies)

        train_features = self._make_features(train)
        self.feature_cols_ = [col for col in train_features.columns if col != "rating"]

        numeric_cols = self.feature_cols_
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            ],
            remainder="drop",
        )
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline_.fit(train_features[self.feature_cols_], train_features["rating"])
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        frame = pd.DataFrame({"user_id": [user_id], "movie_id": [movie_id]})
        return float(self.predict_batch(frame)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before predict_batch().")
        features = self._make_features(test)
        preds = self.pipeline_.predict(features[self.feature_cols_])
        return clip_predictions(np.asarray(preds, dtype=float), *RATING_SCALE)

    def _build_user_features(self, train: pd.DataFrame) -> pd.DataFrame:
        user_stats = train.groupby("user_id").agg(
            user_mean=("rating", "mean"),
            user_std=("rating", "std"),
            user_count=("rating", "size"),
            user_min=("rating", "min"),
            user_max=("rating", "max"),
        )
        user_stats["user_std"] = user_stats["user_std"].fillna(0)
        user_stats["user_log_count"] = np.log1p(user_stats["user_count"])
        return user_stats.reset_index()

    def _build_movie_features(self, train: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
        movie_stats = train.groupby("movie_id").agg(
            item_mean=("rating", "mean"),
            item_std=("rating", "std"),
            item_count=("rating", "size"),
        )
        movie_stats["item_std"] = movie_stats["item_std"].fillna(0)
        movie_stats["item_log_count"] = np.log1p(movie_stats["item_count"])

        movie_meta = movies[["movie_id", "title", "release_date", *self.genre_cols_]].copy()
        movie_meta["release_year"] = movie_meta.apply(_extract_year, axis=1)
        movie_meta["genre_count"] = movie_meta[self.genre_cols_].sum(axis=1)
        movie_meta = movie_meta.drop(columns=["title", "release_date"])
        return movie_meta.merge(movie_stats.reset_index(), on="movie_id", how="left")

    def _make_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        if self.user_features_ is None or self.movie_features_ is None:
            raise RuntimeError("Call fit() before making features.")

        base = ratings[["user_id", "movie_id"]].copy()
        if "rating" in ratings.columns:
            base["rating"] = ratings["rating"].to_numpy()

        out = base.merge(self.user_features_, on="user_id", how="left")
        out = out.merge(self.movie_features_, on="movie_id", how="left")

        defaults = {
            "user_mean": self.global_mean_,
            "user_std": 0.0,
            "user_count": 0.0,
            "user_min": self.global_mean_,
            "user_max": self.global_mean_,
            "user_log_count": 0.0,
            "item_mean": self.global_mean_,
            "item_std": 0.0,
            "item_count": 0.0,
            "item_log_count": 0.0,
            "release_year": out["release_year"].median() if "release_year" in out else np.nan,
            "genre_count": 0.0,
        }
        for col, value in defaults.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        for col in self.genre_cols_:
            out[col] = out[col].fillna(0)

        out["user_item_mean_gap"] = out["user_mean"] - out["item_mean"]
        out["activity_popularity"] = out["user_log_count"] * out["item_log_count"]
        return out


def _extract_year(row: pd.Series) -> float:
    release_date = row.get("release_date")
    if isinstance(release_date, str) and len(release_date) >= 4:
        match = re.search(r"(\d{4})", release_date)
        if match:
            return float(match.group(1))

    title = row.get("title")
    if isinstance(title, str):
        match = re.search(r"\((\d{4})\)", title)
        if match:
            return float(match.group(1))
    return np.nan
