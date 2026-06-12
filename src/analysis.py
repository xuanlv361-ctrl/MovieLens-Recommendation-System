"""Diagnostic analysis helpers for recommendation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae, rmse

NON_GENRE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
}


def prediction_frame(test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Attach predictions, residuals, and absolute errors to a test split."""
    out = test.copy()
    out["prediction"] = np.asarray(predictions, dtype=float)
    out["residual"] = out["rating"] - out["prediction"]
    out["abs_error"] = out["residual"].abs()
    return out


def error_summary(frame: pd.DataFrame) -> dict[str, float]:
    """RMSE/MAE summary for a prediction frame."""
    if frame.empty:
        return {"n": 0, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(frame)),
        "rmse": rmse(frame["rating"].to_numpy(), frame["prediction"].to_numpy()),
        "mae": mae(frame["rating"].to_numpy(), frame["prediction"].to_numpy()),
    }


def cold_start_error(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Compare errors for known vs cold-start users/items in the test split."""
    frame = prediction_frame(test, predictions)
    train_users = set(train["user_id"])
    train_items = set(train["movie_id"])

    frame["user_group"] = np.where(
        frame["user_id"].isin(train_users),
        "known_user",
        "cold_start_user",
    )
    frame["item_group"] = np.where(
        frame["movie_id"].isin(train_items),
        "known_item",
        "cold_start_item",
    )
    frame["case"] = frame["user_group"] + "__" + frame["item_group"]

    rows = []
    for group_name, group in frame.groupby("case"):
        row = {"case": group_name}
        row.update(error_summary(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case").reset_index(drop=True)


def user_activity_error(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    q: int = 4,
) -> pd.DataFrame:
    """Bucket test errors by how many ratings each user has in training."""
    frame = prediction_frame(test, predictions)
    user_counts = train.groupby("user_id").size().rename("train_user_ratings")
    frame = frame.join(user_counts, on="user_id")
    frame["train_user_ratings"] = frame["train_user_ratings"].fillna(0)
    frame["activity_bucket"] = _safe_quantile_bucket(frame["train_user_ratings"], q)
    return _group_error(frame, "activity_bucket")


def movie_popularity_error(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    q: int = 4,
) -> pd.DataFrame:
    """Bucket test errors by how many ratings each movie has in training."""
    frame = prediction_frame(test, predictions)
    item_counts = train.groupby("movie_id").size().rename("train_movie_ratings")
    frame = frame.join(item_counts, on="movie_id")
    frame["train_movie_ratings"] = frame["train_movie_ratings"].fillna(0)
    frame["popularity_bucket"] = _safe_quantile_bucket(frame["train_movie_ratings"], q)
    return _group_error(frame, "popularity_bucket")


def genre_error(
    test: pd.DataFrame,
    movies: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Mean absolute error by movie genre; multi-genre movies count once per genre."""
    frame = prediction_frame(test, predictions)
    genre_cols = [col for col in movies.columns if col not in NON_GENRE_COLUMNS]
    movie_genres = movies[["movie_id", *genre_cols]]
    exploded = frame.merge(movie_genres, on="movie_id", how="left")

    rows = []
    for genre in genre_cols:
        genre_rows = exploded[exploded[genre] == 1]
        if genre_rows.empty:
            continue
        row = {"genre": genre}
        row.update(error_summary(genre_rows))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def tiered_error(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    entity: str = "user",
) -> pd.DataFrame:
    """RMSE/MAE by fixed user-activity or item-popularity tiers."""
    if entity not in {"user", "item"}:
        raise ValueError("entity must be 'user' or 'item'.")

    id_col = "user_id" if entity == "user" else "movie_id"
    count_col = f"train_{entity}_ratings"
    frame = prediction_frame(test, predictions)
    counts = train.groupby(id_col).size().rename(count_col)
    frame = frame.join(counts, on=id_col)
    frame[count_col] = frame[count_col].fillna(0).astype(int)
    frame["tier"] = frame[count_col].map(_activity_tier)
    return _group_error(frame, "tier")


def recommendation_frequency(
    recommendations: dict[int, list[int]],
    all_items: set[int],
) -> pd.DataFrame:
    """Count how often each catalog item appears in recommendation lists."""
    counts = pd.Series(
        [
            item_id
            for user_recs in recommendations.values()
            for item_id in user_recs
        ],
        dtype=int,
    ).value_counts()
    rows = pd.DataFrame({"movie_id": sorted(map(int, all_items))})
    rows["recommendation_count"] = rows["movie_id"].map(counts).fillna(0).astype(int)
    return rows


def recommendation_diversity(
    recommendations: dict[int, list[int]],
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Compute simple per-user genre diversity for recommendation lists."""
    genre_cols = [col for col in movies.columns if col not in NON_GENRE_COLUMNS]
    movie_genres = movies.set_index("movie_id")[genre_cols]
    rows = []
    for user_id, recs in recommendations.items():
        if not recs:
            rows.append(
                {
                    "user_id": user_id,
                    "n_recommendations": 0,
                    "unique_genres": 0,
                    "genre_diversity": 0.0,
                }
            )
            continue
        available = [movie_id for movie_id in recs if movie_id in movie_genres.index]
        if not available:
            unique_genres = 0
        else:
            unique_genres = int((movie_genres.loc[available].sum(axis=0) > 0).sum())
        rows.append(
            {
                "user_id": user_id,
                "n_recommendations": len(recs),
                "unique_genres": unique_genres,
                "genre_diversity": unique_genres / max(1, len(genre_cols)),
            }
        )
    return pd.DataFrame(rows)


def recommendation_novelty(
    recommendations: dict[int, list[int]],
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Compute popularity-based novelty for recommendation lists."""
    item_counts = train.groupby("movie_id").size()
    n_users = train["user_id"].nunique()
    rows = []
    for user_id, recs in recommendations.items():
        if not recs:
            rows.append({"user_id": user_id, "mean_self_information": 0.0})
            continue
        values = []
        for movie_id in recs:
            probability = item_counts.get(movie_id, 0) / n_users
            values.append(-np.log2(max(probability, 1 / n_users)))
        rows.append(
            {
                "user_id": user_id,
                "mean_self_information": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def random_recommend_batch(
    train: pd.DataFrame,
    user_ids: list[int],
    all_items: set[int],
    k: int = 10,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Randomly recommend unseen items for coverage/relevance comparison."""
    rng = np.random.default_rng(random_state)
    user_history = (
        train.groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )
    sorted_items = np.array(sorted(map(int, all_items)))
    recommendations = {}
    for user_id in user_ids:
        seen = user_history.get(int(user_id), set())
        candidates = np.array([item_id for item_id in sorted_items if item_id not in seen])
        if len(candidates) == 0:
            recommendations[int(user_id)] = []
            continue
        size = min(k, len(candidates))
        recommendations[int(user_id)] = rng.choice(
            candidates,
            size=size,
            replace=False,
        ).astype(int).tolist()
    return recommendations


def residual_bias_by_rating(test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Mean prediction bias by actual rating level."""
    frame = prediction_frame(test, predictions)
    frame["prediction_error"] = frame["prediction"] - frame["rating"]
    return (
        frame.groupby("rating")
        .agg(
            n=("rating", "size"),
            mean_prediction=("prediction", "mean"),
            mean_error=("prediction_error", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )


def bootstrap_rmse_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 300,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap a 95% confidence interval for RMSE."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(rmse(y_true[idx], y_pred[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_error_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, float]:
    """Paired t-test on squared errors of two models."""
    from scipy import stats

    err_a = (np.asarray(y_true) - np.asarray(pred_a)) ** 2
    err_b = (np.asarray(y_true) - np.asarray(pred_b)) ** 2
    diff = err_a - err_b
    t_stat, p_value = stats.ttest_rel(err_a, err_b)
    effect = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    return {
        "mean_squared_error_diff": float(diff.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d_paired": float(effect),
    }


def relevant_items_by_user(
    ratings: pd.DataFrame,
    threshold: float = 4.0,
) -> dict[int, set[int]]:
    """Build held-out relevant item sets for top-k evaluation."""
    relevant = ratings[ratings["rating"] >= threshold]
    return (
        relevant.groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )


def recommend_top_k_from_model(
    model,
    train: pd.DataFrame,
    user_ids: list[int],
    all_items: set[int],
    k: int = 10,
    max_users: int | None = 200,
    candidate_items_by_user: dict[int, list[int]] | None = None,
) -> dict[int, list[int]]:
    """
    Build top-k recommendations by scoring candidate items with a fitted model.

    This is intentionally capped by ``max_users`` by default so notebook runtime
    stays reasonable on laptops. Pass ``max_users=None`` for full-catalog
    evaluation when using fast vectorized models.

    Ties are broken with a deterministic popularity prior. This matters for
    constant or near-constant predictors such as GlobalMean/UserMean; otherwise
    equal scores can degenerate into arbitrary movie-ID ordering.
    """
    selected_users = user_ids[:max_users] if max_users is not None else user_ids
    user_history = (
        train.groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )
    popularity = (
        train.groupby("movie_id")
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
    )
    popularity["popularity_score"] = (
        popularity["rating_count"].rank(method="average")
        + popularity["mean_rating"].rank(method="average") / 10000.0
    )
    popularity_score = popularity["popularity_score"].to_dict()

    recommendations: dict[int, list[int]] = {}
    sorted_items = sorted(map(int, all_items))
    for user_id in selected_users:
        seen = user_history.get(int(user_id), set())
        if candidate_items_by_user is None:
            candidates = [item_id for item_id in sorted_items if item_id not in seen]
        else:
            candidates = [
                int(item_id)
                for item_id in candidate_items_by_user.get(int(user_id), [])
                if int(item_id) not in seen
            ]
        if not candidates:
            recommendations[int(user_id)] = []
            continue

        candidate_frame = pd.DataFrame(
            {
                "user_id": int(user_id),
                "movie_id": candidates,
            }
        )
        scores = model.predict_batch(candidate_frame)
        pop_scores = np.array(
            [float(popularity_score.get(item_id, 0.0)) for item_id in candidates],
            dtype=float,
        )
        item_ids = np.array(candidates, dtype=int)
        order = np.lexsort((item_ids, pop_scores, scores))[::-1][:k]
        recommendations[int(user_id)] = [int(candidates[idx]) for idx in order]
    return recommendations


def sampled_candidate_items_by_user(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_ids: list[int],
    all_items: set[int],
    threshold: float = 4.0,
    n_negatives: int = 100,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """
    Build sampled-negative candidate sets for offline top-k evaluation.

    Each candidate set contains the user's held-out relevant items plus a
    deterministic sample of unrated/unseen items. Unrated items are still not
    true negatives; this is a lower-noise benchmark for comparing rankers.
    """
    rng = np.random.default_rng(random_state)
    sorted_items = np.array(sorted(map(int, all_items)), dtype=int)
    train_history = (
        train.groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )
    relevant = relevant_items_by_user(test, threshold=threshold)

    candidates_by_user: dict[int, list[int]] = {}
    for user_id in user_ids:
        user_id = int(user_id)
        positives = set(relevant.get(user_id, set()))
        seen = train_history.get(user_id, set())
        negative_pool = [
            int(item_id)
            for item_id in sorted_items
            if int(item_id) not in seen and int(item_id) not in positives
        ]
        sample_size = min(n_negatives, len(negative_pool))
        if sample_size:
            sampled = rng.choice(negative_pool, size=sample_size, replace=False)
            negatives = [int(item_id) for item_id in sampled]
        else:
            negatives = []
        candidates_by_user[user_id] = sorted(map(int, positives)) + negatives
    return candidates_by_user


def _group_error(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group_name, group in frame.groupby(column, observed=False):
        row = {column: str(group_name)}
        row.update(error_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_quantile_bucket(values: pd.Series, q: int) -> pd.Series:
    ranks = values.rank(method="first")
    try:
        return pd.qcut(ranks, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)


def _activity_tier(count: int) -> str:
    if count <= 3:
        return "very_cold_0_3"
    if count <= 10:
        return "cold_4_10"
    if count <= 50:
        return "warm_11_50"
    return "very_warm_51_plus"
