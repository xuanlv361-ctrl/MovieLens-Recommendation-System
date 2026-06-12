"""Tests for src/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    catalog_coverage,
    clip_predictions,
    evaluate_topk,
    hit_rate_at_k,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_rmse_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_and_mae_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 3.0, 5.0])
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.5))


def test_clip_predictions_bounds():
    preds = np.array([0.0, 3.0, 6.0])
    clipped = clip_predictions(preds, low=1.0, high=5.0)
    assert clipped.tolist() == [1.0, 3.0, 5.0]


def test_precision_recall_hit_rate_at_k():
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 50, 99}
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(2 / 5)
    assert recall_at_k(recommended, relevant, k=5) == pytest.approx(2 / 3)
    assert hit_rate_at_k(recommended, relevant, k=5) == 1.0
    assert hit_rate_at_k([1, 2, 3], relevant, k=3) == 0.0


def test_precision_at_k_invalid_k_raises():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, k=0)


def test_ndcg_at_k_perfect_ranking_is_one():
    recommended = [1, 2, 3]
    relevant = {1, 2}
    assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_items_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_catalog_coverage():
    recs = [[1, 2], [2, 3]]
    all_items = {1, 2, 3, 4}
    assert catalog_coverage(recs, all_items) == pytest.approx(3 / 4)
    assert catalog_coverage(recs, set()) == 0.0


def test_evaluate_topk_aggregates_metrics():
    recommendations = {1: [10, 20, 30], 2: [40, 50, 60]}
    relevant_items = {1: {20}, 2: set()}
    result = evaluate_topk(recommendations, relevant_items, k=3)
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["hit_rate_at_k"] == pytest.approx(1.0)


def test_evaluate_topk_no_users_with_relevant_items():
    result = evaluate_topk({1: [1, 2]}, {1: set()}, k=3)
    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "hit_rate_at_k": 0.0,
        "ndcg_at_k": 0.0,
    }
