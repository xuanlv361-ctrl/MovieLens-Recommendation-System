"""Tests for src/user_based_cf.py and src/item_based_cf.py."""

from __future__ import annotations

import numpy as np

from src.item_based_cf import ItemBasedCF
from src.preprocessing import user_temporal_split
from src.user_based_cf import UserBasedCF


def test_user_based_cf_predictions_within_rating_scale(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    model = UserBasedCF(k=5).fit(train)
    preds = model.predict_batch(test)
    assert preds.shape[0] == len(test)
    assert np.all(preds >= 1.0) and np.all(preds <= 5.0)


def test_user_based_cf_unknown_user_falls_back_to_global_mean(synthetic_ratings):
    train, _ = user_temporal_split(synthetic_ratings)
    model = UserBasedCF(k=5).fit(train)
    pred = model.predict(user_id=999_999, movie_id=1)
    assert pred == model._global_mean


def test_item_based_cf_predictions_within_rating_scale(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    model = ItemBasedCF(k=5).fit(train)
    preds = model.predict_batch(test)
    assert preds.shape[0] == len(test)
    assert np.all(preds >= 1.0) and np.all(preds <= 5.0)


def test_item_based_cf_unknown_movie_falls_back_to_global_mean(synthetic_ratings):
    train, _ = user_temporal_split(synthetic_ratings)
    model = ItemBasedCF(k=5).fit(train)
    pred = model.predict(user_id=1, movie_id=999_999)
    assert pred == model._global_mean
