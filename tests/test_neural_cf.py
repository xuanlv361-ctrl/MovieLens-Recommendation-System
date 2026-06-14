"""Tests for the PyTorch NeuralCF recommender (skipped when torch is unavailable)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.neural_cf import NeuralCFRecommender


def _toy_ratings(n_users=20, n_items=15, seed=0):
    """Build a small synthetic ratings frame for fast tests."""
    rng = np.random.RandomState(seed)
    rows = []
    ts = 0
    for u in range(1, n_users + 1):
        for i in range(1, n_items + 1):
            if rng.rand() < 0.5:
                ts += 1
                rows.append((u, i, int(rng.randint(1, 6)), ts))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def test_fit_records_loss_history():
    """fit() should train for the requested epochs and record a loss per epoch."""
    model = NeuralCFRecommender(epochs=3, batch_size=32, random_state=42)
    model.fit(_toy_ratings())
    assert len(model.history_) == 3
    assert all(np.isfinite(model.history_))


def test_predictions_within_rating_scale():
    """Predicted ratings must be clipped to the valid [1, 5] range."""
    train = _toy_ratings()
    model = NeuralCFRecommender(epochs=2, batch_size=32, random_state=42).fit(train)
    preds = model.predict_batch(train)
    assert preds.min() >= 1.0
    assert preds.max() <= 5.0
    assert len(preds) == len(train)


def test_predict_for_user_returns_series():
    """predict_for_user should return a Series indexed by the candidate movie ids."""
    train = _toy_ratings()
    model = NeuralCFRecommender(epochs=2, batch_size=32, random_state=42).fit(train)
    candidates = [1, 2, 3]
    scores = model.predict_for_user(1, candidates)
    assert list(scores.index) == candidates
    assert len(scores) == 3


def test_fit_writes_experiment_logs(tmp_path):
    """fit(experiment_dir=...) should record config.json and per-epoch metrics.json."""
    import json

    model = NeuralCFRecommender(epochs=3, batch_size=32, random_state=42)
    model.fit(_toy_ratings(), experiment_dir=str(tmp_path))

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.json").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics) == 3
    assert all("train_loss" in v for v in metrics.values())


def test_trainer_validate_returns_loss():
    """NCFTrainer.validate should return a finite mean loss when a val_loader is given."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.trainer import NCFTrainer

    df = _toy_ratings()
    model = NeuralCFRecommender(epochs=1, batch_size=16, random_state=42).fit(df)
    u = torch.tensor([model.user_idx_[int(x)] for x in df["user_id"]], dtype=torch.long)
    i = torch.tensor([model.item_idx_[int(x)] for x in df["movie_id"]], dtype=torch.long)
    r = torch.tensor(df["rating"].to_numpy("float32"))
    loader = DataLoader(TensorDataset(u, i, r), batch_size=16)
    trainer = NCFTrainer(
        model.model_, loader, torch.nn.MSELoss(), torch.optim.AdamW(model.model_.parameters()),
        torch.device("cpu"), epochs=1, val_loader=loader,
    )
    val_loss = trainer.validate()
    assert np.isfinite(val_loss)


def test_save_and_load_roundtrip(tmp_path):
    """A loaded checkpoint should reproduce the original model's predictions."""
    train = _toy_ratings()
    model = NeuralCFRecommender(epochs=2, batch_size=32, random_state=42).fit(train)
    path = tmp_path / "ncf.pt"
    model.save(str(path))

    loaded = NeuralCFRecommender.load(str(path))
    a = model.predict_batch(train)
    b = loaded.predict_batch(train)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
