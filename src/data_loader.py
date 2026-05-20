"""Load MovieLens 100K ratings and movie metadata."""

from __future__ import annotations

import pandas as pd

from .config import DATA_RAW, GENRE_FILE, MOVIES_FILE, RATINGS_FILE


def _check_dataset() -> None:
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"MovieLens 100K not found at {DATA_RAW}.\n"
            "Download from https://grouplens.org/datasets/movielens/100k/ "
            "and extract so that u.data is at data/raw/ml-100k/u.data"
        )


def load_ratings() -> pd.DataFrame:
    """Load u.data: user_id, movie_id, rating, timestamp."""
    _check_dataset()
    df = pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def load_genres() -> list[str]:
    """Load genre names from u.genre (pipe-separated, last line may be empty)."""
    if not GENRE_FILE.exists():
        return []
    genres: list[str] = []
    with open(GENRE_FILE, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line.split("|")[0]
            genres.append(name)
    return genres


def load_movies() -> pd.DataFrame:
    """
    Load u.item: movie_id, title, release_date, video_release_date, imdb_url,
    then 19 binary genre columns.
    """
    _check_dataset()
    genre_names = load_genres()
    n_genres = len(genre_names) if genre_names else 19

    base_cols = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ]
    genre_cols = genre_names if genre_names else [f"genre_{i}" for i in range(n_genres)]
    colnames = base_cols + genre_cols

    movies = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        header=None,
        names=colnames,
        encoding="latin-1",
    )
    # Title in u.item often ends with year in parentheses; keep as-is
    return movies


def dataset_summary(ratings: pd.DataFrame) -> dict:
    """Return basic dataset statistics for EDA."""
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_movies)
    return {
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_movies": n_movies,
        "sparsity": sparsity,
        "min_rating": float(ratings["rating"].min()),
        "max_rating": float(ratings["rating"].max()),
        "mean_rating": float(ratings["rating"].mean()),
        "std_rating": float(ratings["rating"].std()),
    }
