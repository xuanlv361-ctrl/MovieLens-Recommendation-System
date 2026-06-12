"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """A small but dense user-item rating table with many timestamps per user.

    20 users x 15 movies, every pair rated, so per-user temporal splits and
    collaborative-filtering models have enough overlap to be meaningful.
    """
    rng = np.random.default_rng(42)
    rows = []
    timestamp = 1_000_000
    for user_id in range(1, 21):
        for movie_id in range(1, 16):
            rating = int(rng.integers(1, 6))
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    return pd.DataFrame(rows)
