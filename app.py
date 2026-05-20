"""
Streamlit demo: similar-movie recommendations via item-based collaborative filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_movies, load_ratings
from src.item_based_cf import ItemBasedCF

TOP_N = 10


@st.cache_data(show_spinner="Loading MovieLens 100K…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = load_ratings()
    movies = load_movies()
    return ratings, movies


@st.cache_resource(show_spinner="Building item–item similarity (first run may take a few minutes)…")
def load_item_cf(ratings: pd.DataFrame) -> ItemBasedCF:
    model = ItemBasedCF()
    model.fit(ratings)
    return model


@st.cache_data
def movie_avg_ratings(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("movie_id")["rating"].mean()


def build_title_options(movies: pd.DataFrame) -> pd.DataFrame:
    """Sorted lookup: display label -> movie_id."""
    opts = movies[["movie_id", "title"]].copy()
    opts = opts.sort_values("title", key=lambda s: s.str.lower())
    opts["label"] = opts["title"]
    return opts


def recommendations_table(
    similar: pd.Series,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
) -> pd.DataFrame:
    rows = []
    for rank, (movie_id, similarity) in enumerate(similar.items(), start=1):
        title = movies.loc[movies["movie_id"] == movie_id, "title"]
        title_str = title.iloc[0] if len(title) else "Unknown"
        rows.append(
            {
                "Rank": rank,
                "Movie ID": int(movie_id),
                "Title": title_str,
                "Similarity": round(float(similarity), 4),
                "Avg Rating": round(float(avg_ratings.get(movie_id, float("nan"))), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="MovieLens Similar Movies",
        page_icon="🎬",
        layout="wide",
    )

    st.title("MovieLens 100K — Similar Movie Finder")
    st.markdown(
        """
        **University data-mining project** — recommendation on the
        [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset
        (100,000 ratings, 943 users, 1,682 movies).

        **Best model (hold-out evaluation, `random_state=42`):** Item-Based Collaborative
        Filtering — **RMSE 0.9381**, **MAE 0.7342**, outperforming user-based CF, a global-mean
        baseline, and sklearn `TruncatedSVD`.

        This demo uses **item–item cosine similarity** on mean-centered ratings (the same
        engine as Item-Based CF) to suggest movies co-rated by similar audiences.
        """
    )

    try:
        ratings, movies = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Extract ml-100k so that `data/raw/ml-100k/u.data` exists, then restart the app."
        )
        return

    item_cf = load_item_cf(ratings)
    avg_ratings = movie_avg_ratings(ratings)
    title_opts = build_title_options(movies)

    st.sidebar.header("Select a movie")
    selected_label = st.sidebar.selectbox(
        "Movie title",
        options=title_opts["label"].tolist(),
        index=0,
    )
    selected_id = int(
        title_opts.loc[title_opts["label"] == selected_label, "movie_id"].iloc[0]
    )

    st.subheader(f"Similar to: {selected_label}")
    st.caption(f"Movie ID: {selected_id}")

    similar = item_cf.similar_items(selected_id, n=TOP_N)

    if similar.empty:
        st.warning("No similar movies found (insufficient overlap with other items).")
        return

    recs = recommendations_table(similar, movies, avg_ratings)
    st.dataframe(recs, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        **How it works:** Items are compared using **{item_cf.metric} similarity** on
        user rating vectors (minimum {item_cf.min_common} co-ratings). The top **{TOP_N}**
        neighbors with positive similarity are shown. *Avg Rating* is the mean score in the
        full dataset (for context only).
        """
    )


if __name__ == "__main__":
    main()
