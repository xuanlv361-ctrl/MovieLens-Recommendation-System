"""Tests for src/svd_model.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import user_temporal_split
from src.svd_model import SVDRecommender


def test_svd_predictions_within_rating_scale(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    model = SVDRecommender(n_components=5).fit(train)
    preds = model.predict_batch(test)
    assert preds.shape[0] == len(test)
    assert np.all(preds >= 1.0) and np.all(preds <= 5.0)


def test_svd_predict_matches_predict_batch(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    model = SVDRecommender(n_components=5).fit(train)
    batch_preds = model.predict_batch(test)
    single_preds = np.array(
        [model.predict(int(r.user_id), int(r.movie_id)) for r in test.itertuples(index=False)]
    )
    assert np.allclose(batch_preds, single_preds)


def test_svd_unknown_user_and_movie_falls_back_to_baseline(synthetic_ratings):
    train, _ = user_temporal_split(synthetic_ratings)
    model = SVDRecommender(n_components=5).fit(train)
    pred = model.predict(user_id=999_999, movie_id=999_999)
    baseline = model._baseline.predict(999_999, 999_999)
    assert pred == pytest.approx(baseline)


def test_svd_predict_before_fit_raises():
    model = SVDRecommender(n_components=5)
    with pytest.raises(RuntimeError):
        model.predict(1, 1)
