"""Train/test split and rating-matrix utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random hold-out split on rating rows."""
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def global_mean(train: pd.DataFrame) -> float:
    return float(train["rating"].mean())


def user_mean_ratings(train: pd.DataFrame) -> pd.Series:
    return train.groupby("user_id")["rating"].mean()


def item_mean_ratings(train: pd.DataFrame) -> pd.Series:
    return train.groupby("movie_id")["rating"].mean()


def build_user_item_matrix(
    train: pd.DataFrame,
    fill_value: float = np.nan,
) -> pd.DataFrame:
    """Pivot to users (rows) × movies (columns)."""
    return train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=fill_value,
    )


def mean_center_by_user(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract per-user mean (ignoring NaN)."""
    user_means = matrix.mean(axis=1)
    centered = matrix.sub(user_means, axis=0)
    return centered, user_means


def mean_center_by_item(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract per-item mean (ignoring NaN)."""
    item_means = matrix.mean(axis=0)
    centered = matrix.sub(item_means, axis=0)
    return centered, item_means
