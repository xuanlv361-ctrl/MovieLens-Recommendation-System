"""Similarity functions for collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (NaN treated as 0 overlap)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(1.0 - cosine(va, vb))


def pearson_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation similarity between two rating vectors."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.std(va) == 0 or np.std(vb) == 0:
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])


def pairwise_similarity_matrix(
    matrix: pd.DataFrame,
    metric: str = "cosine",
    min_common: int = 1,
) -> pd.DataFrame:
    """
    Compute similarity between all rows of a centered rating matrix.
    Only pairs with at least min_common co-rated entries get non-zero similarity.
    """
    if metric not in ("cosine", "pearson"):
        raise ValueError(f"Unknown metric: {metric}")

    sim_fn = cosine_similarity_vectors if metric == "cosine" else pearson_similarity_vectors
    ids = matrix.index.to_list()
    n = len(ids)
    values = np.zeros((n, n))

    arr = matrix.values
    not_nan = ~np.isnan(arr)

    for i in range(n):
        values[i, i] = 1.0
        for j in range(i + 1, n):
            common = not_nan[i] & not_nan[j]
            if common.sum() < min_common:
                s = 0.0
            else:
                s = sim_fn(arr[i], arr[j])
            values[i, j] = s
            values[j, i] = s

    return pd.DataFrame(values, index=ids, columns=ids)
