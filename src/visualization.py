"""Plotting helpers — figures saved to figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def save_current_figure(filename: str, dpi: int = 150) -> Path:
    out = _ensure_figures_dir() / filename
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out


def plot_rating_distribution(ratings: pd.DataFrame, filename: str = "rating_distribution.png") -> Path:
    fig, ax = plt.subplots()
    sns.histplot(ratings["rating"], bins=5, discrete=True, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings (MovieLens 100K)")
    return save_current_figure(filename)


def plot_ratings_per_user(ratings: pd.DataFrame, filename: str = "ratings_per_user.png") -> Path:
    counts = ratings.groupby("user_id").size()
    fig, ax = plt.subplots()
    sns.histplot(counts, bins=50, log_scale=True, ax=ax)
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of users")
    ax.set_title("Ratings per User (log scale)")
    return save_current_figure(filename)


def plot_ratings_per_movie(ratings: pd.DataFrame, filename: str = "ratings_per_movie.png") -> Path:
    counts = ratings.groupby("movie_id").size()
    fig, ax = plt.subplots()
    sns.histplot(counts, bins=50, log_scale=True, ax=ax)
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of movies")
    ax.set_title("Ratings per Movie (log scale)")
    return save_current_figure(filename)


def plot_genre_counts(movies: pd.DataFrame, filename: str = "genre_counts.png") -> Path:
    genre_cols = [c for c in movies.columns if c not in (
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    )]
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
    """results: columns model, rmse, mae."""
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
    ax.set_title(f"Predicted vs Actual — {model_name}")
    ax.legend()
    return save_current_figure(filename)


def plot_itemcf_k_sensitivity(
    k_results: pd.DataFrame,
    filename: str = "itemcf_k_sensitivity.png",
) -> Path:
    """Line plot of RMSE and MAE vs k for Item-Based CF."""
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


def plot_error_by_rating(
    error_by_rating: pd.Series,
    model_name: str,
    filename: str | None = None,
) -> Path:
    if filename is None:
        safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"error_by_rating_{safe}.png"

    fig, ax = plt.subplots()
    error_by_rating.sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Prediction Error by Rating Level — {model_name}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    return save_current_figure(filename)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str | None = None,
) -> Path:
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"residuals_{safe}.png"

    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=40, kde=True, ax=ax)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_title(f"Residual Distribution — {model_name}")
    return save_current_figure(filename)
