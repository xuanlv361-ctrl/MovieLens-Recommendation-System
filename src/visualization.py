"""Plotting helpers; figures are saved to figures/."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)

NON_GENRE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
}


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def save_current_figure(filename: str, dpi: int = 150) -> Path:
    """Save the current matplotlib figure to the figures directory."""
    out = _ensure_figures_dir() / filename
    plt.tight_layout()
    try:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
    except PermissionError:
        fallback = out.with_name(f"{out.stem}_{int(time.time())}{out.suffix}")
        try:
            plt.savefig(fallback, dpi=dpi, bbox_inches="tight")
            out = fallback
        except PermissionError:
            print(f"Warning: could not save figure to {out} or {fallback}; continuing.")
    plt.close()
    return out


def plot_rating_distribution(
    ratings: pd.DataFrame,
    filename: str = "rating_distribution.png",
) -> Path:
    """Plot the histogram of rating values."""
    fig, ax = plt.subplots()
    sns.histplot(ratings["rating"], bins=5, discrete=True, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings (MovieLens 100K)")
    return save_current_figure(filename)


def plot_ratings_per_user(
    ratings: pd.DataFrame,
    filename: str = "ratings_per_user.png",
) -> Path:
    """Plot the distribution of the number of ratings per user."""
    counts = ratings.groupby("user_id").size()
    fig, ax = plt.subplots()
    sns.histplot(counts, bins=50, log_scale=True, ax=ax)
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of users")
    ax.set_title("Ratings per User (log scale)")
    return save_current_figure(filename)


def plot_ratings_per_movie(
    ratings: pd.DataFrame,
    filename: str = "ratings_per_movie.png",
) -> Path:
    """Plot the distribution of the number of ratings per movie."""
    counts = ratings.groupby("movie_id").size()
    fig, ax = plt.subplots()
    sns.histplot(counts, bins=50, log_scale=True, ax=ax)
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of movies")
    ax.set_title("Ratings per Movie (log scale)")
    return save_current_figure(filename)


def plot_genre_counts(
    movies: pd.DataFrame,
    filename: str = "genre_counts.png",
) -> Path:
    """Plot the number of movies per genre."""
    genre_cols = [col for col in movies.columns if col not in NON_GENRE_COLUMNS]
    if not genre_cols:
        return _ensure_figures_dir() / filename

    counts = movies[genre_cols].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of movies")
    ax.set_title("Movies per Genre")
    return save_current_figure(filename)


def plot_model_comparison(
    results: pd.DataFrame,
    filename: str = "model_comparison.png",
) -> Path:
    """Plot RMSE and MAE bars for a model comparison table."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=results, x="model", y="rmse", ax=axes[0], palette="Blues_d")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=15)
    sns.barplot(data=results, x="model", y="mae", ax=axes[1], palette="Greens_d")
    axes[1].set_title("MAE (lower is better)")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Model Comparison on Hold-out Test Set")
    return save_current_figure(filename)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str | None = None,
    max_points: int = 5000,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Plot predicted ratings against actual ratings for a model."""
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"pred_vs_actual_{safe}.png"

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    idx = rng.choice(n, size=min(max_points, n), replace=False)

    fig, ax = plt.subplots()
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=10)
    ax.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Predicted rating")
    ax.set_title(f"Predicted vs Actual - {model_name}")
    ax.legend()
    return save_current_figure(filename)


def plot_k_sensitivity(
    k_results: pd.DataFrame,
    filename: str = "k_sensitivity.png",
) -> Path:
    """Line plot of RMSE and MAE vs k for CF models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in k_results.groupby("model"):
        ax.plot(
            group["k"],
            group["rmse"],
            marker="o",
            linewidth=2,
            label=f"{model_name} RMSE",
        )
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("RMSE")
    ax.set_title("Collaborative Filtering Sensitivity to K_NEIGHBORS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_current_figure(filename)


def plot_itemcf_k_sensitivity(
    k_results: pd.DataFrame,
    filename: str = "itemcf_k_sensitivity.png",
) -> Path:
    """Backward-compatible item-CF k sensitivity plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_results["k"], k_results["rmse"], marker="o", linewidth=2, label="RMSE")
    ax.plot(k_results["k"], k_results["mae"], marker="s", linewidth=2, label="MAE")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Error")
    ax.set_title("Item-Based CF: Sensitivity to K_NEIGHBORS")
    ax.set_xticks(k_results["k"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_current_figure(filename)


def plot_svd_sensitivity(
    svd_results: pd.DataFrame,
    filename: str = "svd_components_sensitivity.png",
) -> Path:
    """Line plot of RMSE and MAE vs SVD latent dimension."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(svd_results["n_components"], svd_results["rmse"], marker="o", label="RMSE")
    ax.plot(svd_results["n_components"], svd_results["mae"], marker="s", label="MAE")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Error")
    ax.set_title("SVD Sensitivity to Number of Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_current_figure(filename)


def plot_bias_regularization(
    bias_results: pd.DataFrame,
    filename: str = "bias_regularization_sensitivity.png",
) -> Path:
    """Plot train/test RMSE over regularization strengths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bias_results["reg"], bias_results["train_rmse"], marker="o", label="Train RMSE")
    ax.plot(bias_results["reg"], bias_results["test_rmse"], marker="s", label="Test RMSE")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength")
    ax.set_ylabel("RMSE")
    ax.set_title("Bias Baseline Regularization Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_current_figure(filename)


def plot_topn_metrics(
    ranking_results: pd.DataFrame,
    filename: str = "topn_metrics_comparison.png",
) -> Path:
    """Plot top-N ranking metrics for each model."""
    metric_cols = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
    long = ranking_results.melt(
        id_vars="model",
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Top-N Ranking Metrics by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric", loc="best")
    return save_current_figure(filename)


def plot_tier_error(
    tier_results: pd.DataFrame,
    filename: str = "tier_error.png",
) -> Path:
    """Plot RMSE by cold/warm user or item tier."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=tier_results, x="tier", y="rmse", hue="model", ax=ax)
    ax.set_title("Error by Activity/Popularity Tier")
    ax.set_xlabel("Tier")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=20)
    return save_current_figure(filename)


def plot_genre_error(
    genre_results: pd.DataFrame,
    filename: str = "genre_rmse.png",
) -> Path:
    """Plot genre-level RMSE values."""
    plot_data = genre_results.sort_values("rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(data=plot_data, y="genre", x="rmse", ax=ax, color="steelblue")
    ax.set_title("RMSE by Movie Genre")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Genre")
    return save_current_figure(filename)


def plot_recommendation_frequency(
    frequency: pd.DataFrame,
    filename: str = "recommendation_frequency.png",
) -> Path:
    """Plot how often catalog items appear in recommendation lists."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(frequency["recommendation_count"], bins=30, ax=ax)
    ax.set_title("Recommendation Frequency Across Catalog")
    ax.set_xlabel("Times recommended")
    ax.set_ylabel("Number of movies")
    return save_current_figure(filename)


def plot_learning_curve(
    learning_results: pd.DataFrame,
    filename: str = "learning_curve.png",
) -> Path:
    """Plot train/test RMSE as training-set fraction changes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=learning_results,
        x="train_fraction",
        y="rmse",
        hue="model",
        style="split",
        marker="o",
        ax=ax,
    )
    ax.set_title("Learning Curves")
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    return save_current_figure(filename)


def plot_diversity_novelty(
    diversity_results: pd.DataFrame,
    filename: str = "diversity_novelty.png",
) -> Path:
    """Plot diversity and novelty summaries for top-N recommenders."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=diversity_results, x="model", y="genre_diversity", ax=axes[0])
    axes[0].set_title("Mean Genre Diversity")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(
        data=diversity_results,
        x="model",
        y="mean_self_information",
        ax=axes[1],
    )
    axes[1].set_title("Mean Novelty")
    axes[1].tick_params(axis="x", rotation=20)
    return save_current_figure(filename)


def plot_error_by_rating(
    error_by_rating: pd.Series,
    model_name: str,
    filename: str | None = None,
) -> Path:
    """Plot prediction error grouped by the true rating value."""
    if filename is None:
        safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"error_by_rating_{safe}.png"

    fig, ax = plt.subplots()
    error_by_rating.sort_index().plot(
        kind="bar",
        ax=ax,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Prediction Error by Rating Level - {model_name}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    return save_current_figure(filename)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str | None = None,
) -> Path:
    """Plot the distribution of prediction residuals for a model."""
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"residuals_{safe}.png"

    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=40, kde=True, ax=ax)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_title(f"Residual Distribution - {model_name}")
    return save_current_figure(filename)
