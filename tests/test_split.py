"""Tests for src/preprocessing.py split functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    user_temporal_split,
    user_temporal_train_val_test_split,
)


def test_user_temporal_split_preserves_row_count(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    assert len(train) + len(test) == len(synthetic_ratings)
    assert not train.empty
    assert not test.empty


def test_user_temporal_split_every_test_user_has_training_history(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    train_users = set(train["user_id"])
    test_users = set(test["user_id"])
    assert test_users.issubset(train_users)


def test_user_temporal_split_no_future_leakage_per_user(synthetic_ratings):
    train, test = user_temporal_split(synthetic_ratings)
    for user_id in test["user_id"].unique():
        max_train_ts = train.loc[train["user_id"] == user_id, "timestamp"].max()
        min_test_ts = test.loc[test["user_id"] == user_id, "timestamp"].min()
        assert max_train_ts < min_test_ts


def test_user_temporal_split_missing_columns_raise():
    bad = pd.DataFrame({"user_id": [1, 2], "rating": [3, 4]})
    with pytest.raises(ValueError):
        user_temporal_split(bad)


def test_train_val_test_split_preserves_row_count(synthetic_ratings):
    train, val, test = user_temporal_train_val_test_split(synthetic_ratings)
    assert len(train) + len(val) + len(test) == len(synthetic_ratings)
    assert not train.empty
    assert not val.empty
    assert not test.empty


def test_train_val_test_split_chronological_ordering_per_user(synthetic_ratings):
    train, val, test = user_temporal_train_val_test_split(synthetic_ratings)
    for user_id in synthetic_ratings["user_id"].unique():
        train_ts = train.loc[train["user_id"] == user_id, "timestamp"]
        val_ts = val.loc[val["user_id"] == user_id, "timestamp"]
        test_ts = test.loc[test["user_id"] == user_id, "timestamp"]
        if not train_ts.empty and not val_ts.empty:
            assert train_ts.max() < val_ts.min()
        if not val_ts.empty and not test_ts.empty:
            assert val_ts.max() < test_ts.min()


def test_train_val_test_split_invalid_sizes_raise(synthetic_ratings):
    with pytest.raises(ValueError):
        user_temporal_train_val_test_split(synthetic_ratings, val_size=0.5, test_size=0.6)
