"""Streamlit demo for rating prediction and similar-movie recommendation."""

from __future__ import annotations

import base64
import html
import math
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import accounts, db
from src import recsys_service as recsys
from src.auth import verify_admin
from src.baselines import BiasBaseline
from src.data_cleaning import align_ratings_with_movies, clean_movies, clean_ratings
from src.data_loader import load_movies, load_ratings
from src.hybrid_model import MetadataHybridRegressor
from src.i18n import (
    GENRE_EN_BY_ZH,
    GENRE_ZH,
    ONBOARDING_GENRES,
    T,
    get_display_title,
    translate_columns,
    translate_genres,
)
from src.item_based_cf import ItemBasedCF
from src.metrics import evaluate_topk, mae, rmse
from src.posters import (
    LOCAL_POSTER_DIR,
    build_local_poster_index,
    fetch_poster_url,
    local_poster_path,
    refresh_poster_cache,
)
from src.posters import poster_background as poster_gradient
from src.preprocessing import user_temporal_split
from src.svd_model import SVDRecommender
from src.user_based_cf import UserBasedCF

TOP_N = 10


def render_html_block(template: str) -> None:
    """Render a multi-line HTML/CSS template via st.markdown.

    Streamlit's Markdown renderer treats lines indented by 4+ spaces as a
    literal code block, which causes nested HTML (e.g. multi-line template
    strings with indented child tags) to show up as raw text instead of
    being rendered. Stripping per-line indentation and dropping blank lines
    (which can also terminate an HTML block early) avoids that.
    """
    lines = [line.strip() for line in template.strip().splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


@st.cache_data(show_spinner="正在加载并清洗 MovieLens 100K 数据...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = clean_ratings(load_ratings())
    movies = clean_movies(load_movies())
    ratings, movies = align_ratings_with_movies(ratings, movies)
    return ratings, movies


@st.cache_resource(show_spinner="正在构建电影相似度矩阵...")
def load_item_cf(_ratings: pd.DataFrame) -> ItemBasedCF:
    model = ItemBasedCF()
    model.fit(_ratings)
    return model


@st.cache_resource(show_spinner="正在训练基于用户的协同过滤模型...")
def load_user_cf(_ratings: pd.DataFrame, k: int = 20, metric: str = "cosine") -> UserBasedCF:
    model = UserBasedCF(k=k, metric=metric)
    model.fit(_ratings)
    return model


@st.cache_resource(show_spinner="正在训练基于物品的协同过滤模型...")
def load_item_cf_configured(
    _ratings: pd.DataFrame,
    k: int = 20,
    metric: str = "cosine",
) -> ItemBasedCF:
    model = ItemBasedCF(k=k, metric=metric)
    model.fit(_ratings)
    return model


@st.cache_resource(show_spinner="正在训练 SVD 矩阵分解模型...")
def load_svd_model(_ratings: pd.DataFrame, factors: int = 50) -> SVDRecommender:
    model = SVDRecommender(n_components=factors)
    model.fit(_ratings)
    return model


@st.cache_resource(show_spinner="正在训练评分预测模型...")
def load_bias_model(_ratings: pd.DataFrame) -> BiasBaseline:
    model = BiasBaseline(n_epochs=20, reg=1.0)
    model.fit(_ratings)
    return model


@st.cache_data
def movie_avg_ratings(_ratings: pd.DataFrame) -> pd.Series:
    return _ratings.groupby("movie_id")["rating"].mean()


@st.cache_data
def movie_rating_counts(_ratings: pd.DataFrame) -> pd.Series:
    return _ratings.groupby("movie_id")["rating"].count()


def movie_catalog(
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
) -> pd.DataFrame:
    rows = []
    for row in movies.itertuples(index=False):
        movie_id = int(row.movie_id)
        movie_row = movies.loc[movies["movie_id"] == movie_id].iloc[0]
        rows.append(
            {
                "Movie ID": movie_id,
                "Title": str(row.title),
                "Genres": movie_genre_text(movie_row, movies),
                "Average Rating": round(float(avg_ratings.get(movie_id, np.nan)), 2),
                "Ratings": int(rating_counts.get(movie_id, 0)),
                "Release Year": movie_row.get("release_year", ""),
            }
        )
    return pd.DataFrame(rows)


def search_movies(
    catalog: pd.DataFrame,
    query: str,
    genre: str,
) -> pd.DataFrame:
    """Filter the catalog (English columns) by query and genre.

    - Empty query: no title/ID filtering (caller decides the default Top-N view).
    - Numeric query: exact match on Movie ID.
    - Non-numeric query: case-insensitive substring match on the English title
      or the mapped Chinese display title.
    - `genre` may be the "all genres" sentinel (`T["label_all"]`, i.e. "全部"),
      an English MovieLens genre name, or its Chinese translation.
    """
    result = catalog.copy()
    query = query.strip()
    if query:
        if query.isdigit():
            result = result[result["Movie ID"] == int(query)]
        else:
            query_lower = query.lower()
            title_match = result["Title"].str.lower().str.contains(query_lower, na=False)
            zh_titles = result["Title"].map(get_display_title)
            zh_match = zh_titles.str.contains(query, na=False)
            result = result[title_match | zh_match]
    if genre != T["label_all"]:
        english_genre = GENRE_EN_BY_ZH.get(genre, genre)
        result = result[result["Genres"].str.contains(english_genre, case=False, na=False)]
    return result.sort_values(["Average Rating", "Ratings"], ascending=False)


def popular_movies(
    catalog: pd.DataFrame,
    rank_by: str,
    top_n: int,
) -> pd.DataFrame:
    return recsys.popular_movies(catalog, rank_by, top_n)


def algorithm_model(
    ratings: pd.DataFrame,
    algorithm: str,
    k: int,
    metric: str,
    factors: int,
) -> object:
    if algorithm == "User-based Collaborative Filtering":
        return load_user_cf(ratings, k=k, metric=metric)
    if algorithm == "SVD Matrix Factorization":
        return load_svd_model(ratings, factors=factors)
    return load_item_cf_configured(ratings, k=k, metric=metric)


def recommendation_reason(algorithm: str, genres: str) -> str:
    return recsys.recommendation_reason(algorithm, genres)


def recommendation_explanation(algorithm: str) -> str:
    """Return a Chinese explanation of how the selected algorithm ranks recommendations."""
    explanations = {
        "Item-based Collaborative Filtering": (
            "**Item-based Collaborative Filtering（ItemCF）** "
            "会分析你评分较高的电影，找到与这些电影评分模式相似的其他电影"
            "（基于物品-物品相似度），并按相似度与预测评分排序后推荐给你。"
        ),
        "User-based Collaborative Filtering": (
            "**User-based Collaborative Filtering（UserCF）** "
            "会找到与你评分习惯相似的其他用户（基于用户-用户相似度），"
            "并将这些相似用户喜欢但你还未观看的电影推荐给你。"
        ),
        "SVD Matrix Factorization": (
            "**SVD Matrix Factorization** "
            "通过矩阵分解学习每个用户和每部电影的隐含特征（隐因子），"
            "并用这些隐因子计算你对未观看电影的预测评分，再按预测评分排序推荐。"
        ),
    }
    return explanations.get(
        algorithm,
        "该算法会根据你的历史评分数据计算预测评分，并按预测评分从高到低排序推荐。",
    )


def _catalog_rows_to_recs(rows: pd.DataFrame, catalog: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Reshape catalog rows (Movie ID/Title/Genres/Average Rating/Ratings/Release Year)
    into the recommendation-card column layout produced by `personalized_recommendations`
    (Rank/Movie ID/Title/Genres/Release Year/Average Rating/Ratings/Recommendation Score/
    Confidence/Reason), used for genre-based and popularity cold-start fallbacks."""
    max_count = max(1, int(catalog["Ratings"].max()))
    records = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        rating_count = int(row["Ratings"])
        confidence = min(1.0, math.log1p(rating_count) / math.log1p(max_count))
        records.append(
            {
                "Rank": rank,
                "Movie ID": int(row["Movie ID"]),
                "Title": row["Title"],
                "Genres": row["Genres"],
                "Release Year": row["Release Year"],
                "Average Rating": row["Average Rating"],
                "Ratings": rating_count,
                "Recommendation Score": row["Average Rating"],
                "Confidence": round(confidence, 2),
                "Reason": reason,
            }
        )
    return pd.DataFrame(records)


def genre_based_recommendations(catalog: pd.DataFrame, preferred_genres_zh: list[str], top_n: int = 10) -> pd.DataFrame:
    """Cold-start fallback: rank catalog movies matching the user's preferred genres
    (selected at registration) by average rating then rating count."""
    english_genres = [GENRE_EN_BY_ZH.get(g, g) for g in preferred_genres_zh if g]
    if not english_genres:
        return pd.DataFrame()
    pattern = "|".join(re.escape(g) for g in english_genres)
    matches = catalog[catalog["Genres"].str.contains(pattern, case=False, na=False, regex=True)]
    if matches.empty:
        return pd.DataFrame()
    matches = matches.sort_values(["Average Rating", "Ratings"], ascending=False).head(top_n)
    reason = f"该电影类型符合你注册时选择的偏好：{'、'.join(preferred_genres_zh)}"
    return _catalog_rows_to_recs(matches, catalog, reason)


def apply_genre_preference_boost(recs: pd.DataFrame, preferred_genres_zh: list[str]) -> pd.DataFrame:
    """Re-rank collaborative-filtering recommendations so movies matching the
    user's selected preference genres are listed first, while still keeping
    every original CF candidate (just reordered, with `Rank` renumbered)."""
    if recs.empty or not preferred_genres_zh:
        return recs
    english_genres = [GENRE_EN_BY_ZH.get(g, g) for g in preferred_genres_zh if g]
    if not english_genres:
        return recs
    pattern = "|".join(re.escape(g) for g in english_genres)
    matches = recs["Genres"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
    reordered = pd.concat([recs[matches], recs[~matches]], ignore_index=True)
    reordered["Rank"] = range(1, len(reordered) + 1)
    return reordered


def popularity_recommendations(catalog: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Global-popularity fallback for users without enough ratings or preferences."""
    top = popular_movies(catalog, "Weighted Score", top_n)
    return _catalog_rows_to_recs(top, catalog, "全站热门高分电影推荐")


@st.cache_data(show_spinner=False)
def personalized_recommendations(
    _ratings: pd.DataFrame,
    _movies: pd.DataFrame,
    _avg_ratings: pd.Series,
    _rating_counts: pd.Series,
    user_id: int,
    algorithm: str,
    top_n: int,
    k: int,
    metric: str,
    factors: int,
) -> pd.DataFrame:
    """Score and rank unrated movies for one user.

    All candidate movies are scored in a single vectorized pass via
    `model.predict_for_user(...)` and `calibrated_score_batch(...)`, instead
    of calling `model.predict()` in a Python loop for ~1,600 candidates.
    """
    model = algorithm_model(_ratings, algorithm, k, metric, factors)
    rated = set(_ratings.loc[_ratings["user_id"] == user_id, "movie_id"])
    candidates = [int(mid) for mid in _movies["movie_id"] if int(mid) not in rated]
    if not candidates:
        return pd.DataFrame()

    user_mean = float(_ratings.loc[_ratings["user_id"] == user_id, "rating"].mean())
    global_mean = float(_ratings["rating"].mean())
    max_count = max(1, int(_rating_counts.max()))

    raw_pred_arr = model.predict_for_user(user_id, candidates).reindex(candidates).to_numpy(dtype=float)
    avg_rating_arr = _avg_ratings.reindex(candidates).fillna(global_mean).to_numpy(dtype=float)
    rating_count_arr = _rating_counts.reindex(candidates).fillna(0).to_numpy(dtype=float)
    confidence_arr = np.minimum(1.0, np.log1p(rating_count_arr) / np.log1p(max_count))

    scores = recsys.calibrated_score_batch(
        raw_pred_arr, avg_rating_arr, user_mean, confidence_arr, rating_count_arr, global_mean,
    )

    movies_indexed = _movies.set_index("movie_id")
    genre_cols = display_genre_columns(_movies)
    cand_rows = movies_indexed.loc[candidates]
    genres_list = []
    for _, row in cand_rows.iterrows():
        present = [genre for genre in genre_cols if int(row.get(genre, 0) or 0) == 1]
        genres_list.append(", ".join(present) if present else "未分类")

    recs = pd.DataFrame(
        {
            "Movie ID": candidates,
            "Title": cand_rows["title"].to_numpy(),
            "Recommendation Score": scores,
            "Raw Model Score": raw_pred_arr,
            "Average Rating": avg_rating_arr,
            "Ratings": rating_count_arr.astype(int),
            "Confidence": confidence_arr,
            "Genres": genres_list,
            "Release Year": cand_rows["release_year"].to_numpy(),
            "Reason": [recommendation_reason(algorithm, genres) for genres in genres_list],
        }
    )
    recs = recs.sort_values(
        ["Recommendation Score", "Confidence", "Average Rating", "Ratings"],
        ascending=False,
    ).head(top_n).reset_index(drop=True)
    recs.insert(0, "Rank", range(1, len(recs) + 1))
    recs["Recommendation Score"] = recs["Recommendation Score"].round(2)
    recs["Raw Model Score"] = recs["Raw Model Score"].round(2)
    recs["Average Rating"] = recs["Average Rating"].round(2)
    recs["Confidence"] = recs["Confidence"].round(2)
    return recs


def safe_personalized_recommendations(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    user_id: int,
    algorithm: str,
    top_n: int,
    k: int,
    metric: str,
    factors: int,
) -> tuple[pd.DataFrame, float]:
    """Compute personalized recommendations with a spinner and a Chinese error message.

    Returns the recommendation table and the wall-clock time (seconds) spent
    inside `personalized_recommendations`. On a cache hit this is close to 0.
    """
    try:
        with st.spinner("正在生成个性化推荐..."):
            start = time.perf_counter()
            recs = personalized_recommendations(
                ratings, movies, avg_ratings, rating_counts, user_id, algorithm, top_n, k, metric, factors,
            )
            elapsed = time.perf_counter() - start
        return recs, elapsed
    except Exception as exc:  # noqa: BLE001
        st.error(f"生成推荐时发生错误，请尝试调整参数后重试。错误信息：{exc}")
        return pd.DataFrame(), 0.0


def recommendation_genre_counts(recs: pd.DataFrame) -> pd.Series:
    counts: dict[str, int] = {}
    for genres in recs.get("Genres", []):
        for genre in str(genres).split(","):
            genre = genre.strip()
            if genre and genre != "未分类":
                genre_zh = GENRE_ZH.get(genre, genre)
                counts[genre_zh] = counts.get(genre_zh, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


@st.cache_data(show_spinner="正在采样留出集上评估模型...")
def evaluation_summary(_ratings: pd.DataFrame) -> pd.DataFrame:
    train, test = user_temporal_split(_ratings)
    test_sample = test.sample(n=min(1500, len(test)), random_state=42)
    models = {
        "User-based CF": UserBasedCF(k=20).fit(train),
        "Item-based CF": ItemBasedCF(k=20).fit(train),
        "SVD": SVDRecommender(n_components=50).fit(train),
    }
    rows = []
    y_true = test_sample["rating"].to_numpy(dtype=float)
    for name, model in models.items():
        preds = model.predict_batch(test_sample)
        rows.append(
            {
                "Algorithm": name,
                "MAE": round(mae(y_true, preds), 4),
                "RMSE": round(rmse(y_true, preds), 4),
                "Evaluated Ratings": len(test_sample),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="正在计算模型评估指标（含排序指标，可能需要一些时间）...")
def ranking_evaluation_summary(
    _ratings: pd.DataFrame,
    _movies: pd.DataFrame,
    k: int = 10,
    n_users: int = 60,
) -> pd.DataFrame:
    """Compare UserCF / ItemCF / SVD / Hybrid / Baseline on rating-error and top-k ranking metrics.

    Ranking metrics are computed for a sampled subset of users against a
    candidate pool (popular movies + each user's relevant test items) to keep
    the computation lightweight enough for an admin dashboard. Cached so it
    only runs once per dataset/parameter combination.
    """
    train, test = user_temporal_split(_ratings)
    test_sample = test.sample(n=min(1500, len(test)), random_state=42)
    y_true = test_sample["rating"].to_numpy(dtype=float)

    models: dict[str, object] = {
        "User-based CF": UserBasedCF(k=20).fit(train),
        "Item-based CF": ItemBasedCF(k=20).fit(train),
        "SVD": SVDRecommender(n_components=50).fit(train),
        "Hybrid": MetadataHybridRegressor().fit(train, _movies),
        "Baseline": BiasBaseline(n_epochs=20, reg=1.0).fit(train),
    }

    popular_movie_ids = (
        train.groupby("movie_id")["rating"].count().sort_values(ascending=False).head(300).index.tolist()
    )

    relevant_test = test[test["rating"] >= 4]
    rng = np.random.RandomState(42)
    eval_users = relevant_test["user_id"].unique()
    if len(eval_users) > n_users:
        eval_users = rng.choice(eval_users, size=n_users, replace=False)
    eval_users = [int(u) for u in eval_users]

    relevant_items: dict[int, set[int]] = {
        u: set(relevant_test.loc[relevant_test["user_id"] == u, "movie_id"]) for u in eval_users
    }
    train_seen: dict[int, set[int]] = {
        u: set(train.loc[train["user_id"] == u, "movie_id"]) for u in eval_users
    }

    rows = []
    for name, model in models.items():
        preds = model.predict_batch(test_sample)

        recommendations: dict[int, list[int]] = {}
        for u in eval_users:
            candidates = (set(popular_movie_ids) | relevant_items.get(u, set())) - train_seen.get(u, set())
            candidates = sorted(candidates)
            if hasattr(model, "predict_for_user"):
                scores = model.predict_for_user(u, candidates)
            else:
                cand_df = pd.DataFrame({"user_id": u, "movie_id": candidates})
                scores = pd.Series(model.predict_batch(cand_df), index=candidates)
            recommendations[u] = [int(m) for m in scores.sort_values(ascending=False).head(k).index]

        ranking = evaluate_topk(recommendations, relevant_items, k=k)
        rows.append(
            {
                "Algorithm": name,
                "MAE": round(mae(y_true, preds), 4),
                "RMSE": round(rmse(y_true, preds), 4),
                "Precision@10": round(ranking["precision_at_k"], 4),
                "Recall@10": round(ranking["recall_at_k"], 4),
                "HitRate@10": round(ranking["hit_rate_at_k"], 4),
                "NDCG@10": round(ranking["ndcg_at_k"], 4),
                "Evaluated Ratings": len(test_sample),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def rating_error_interval(
    ratings: pd.DataFrame,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> tuple[float, float, float]:
    """Approximate prediction interval from in-sample residual quantiles."""
    calibration_model = BiasBaseline(n_epochs=20, reg=1.0).fit(ratings)
    preds = calibration_model.predict_batch(ratings)
    errors = ratings["rating"].to_numpy(dtype=float) - preds
    low = float(pd.Series(errors).quantile(lower_quantile))
    high = float(pd.Series(errors).quantile(upper_quantile))
    rmse = float((errors**2).mean() ** 0.5)
    return low, high, rmse


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
        title_str = title.iloc[0] if len(title) else "未知"
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


def movie_genre_text(movie_row: pd.Series, movies: pd.DataFrame) -> str:
    genres = recsys.genre_text(movie_row, display_genre_columns(movies))
    return genres if genres != "Unknown" else "未分类"


def movie_detail(
    movie_id: int,
    recs: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> dict[str, object]:
    movie_row = movies.loc[movies["movie_id"] == movie_id].iloc[0]
    rec_row = recs.loc[recs["Movie ID"] == movie_id].iloc[0]
    movie_ratings = ratings.loc[ratings["movie_id"] == movie_id, "rating"]
    return {
        "title": str(movie_row["title"]),
        "release_year": movie_row.get("release_year", ""),
        "release_date": movie_row.get("release_date", ""),
        "genres": movie_genre_text(movie_row, movies),
        "imdb_url": movie_row.get("imdb_url", ""),
        "rank": int(rec_row["Rank"]),
        "similarity": float(rec_row["Similarity"]),
        "avg_rating": float(rec_row["Avg Rating"]),
        "rating_count": int(movie_ratings.count()),
    }


def build_safe_imdb_url(movie_title: str, imdb_url: str | None = None) -> str:
    """Return a safe HTTPS www.imdb.com URL for `movie_title`.

    MovieLens 100K ships IMDb links on the retired `us.imdb.com` domain,
    which now serves an invalid certificate (`NET::ERR_CERT_COMMON_NAME_INVALID`).
    Rather than open those links directly, extract an IMDb title id (`ttXXXXXXX`)
    if present and build a modern `https://www.imdb.com/title/ttXXXXXXX/` URL;
    otherwise fall back to an IMDb title search for `movie_title`.
    """
    title = str(movie_title or "").strip()

    if imdb_url:
        url = str(imdb_url).strip()
        match = re.search(r"(tt\d+)", url)
        if match:
            return f"https://www.imdb.com/title/{match.group(1)}/"

    return f"https://www.imdb.com/find/?q={quote(title)}&s=tt"


def genre_columns(movies: pd.DataFrame) -> list[str]:
    return recsys.genre_columns(movies)


def display_genre_columns(movies: pd.DataFrame) -> list[str]:
    """Genre columns shown to users, excluding the MovieLens "unknown" placeholder genre."""
    return [genre for genre in genre_columns(movies) if genre.lower() != "unknown"]


def genre_options_zh(movies: pd.DataFrame) -> list[str]:
    """Chinese genre names for filter selectboxes, derived from genre columns."""
    return [GENRE_ZH.get(genre, genre) for genre in display_genre_columns(movies)]


def user_history_table(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    history = ratings.loc[ratings["user_id"] == user_id].merge(
        movies[["movie_id", "title", "release_year"]],
        on="movie_id",
        how="left",
    )
    history = history.sort_values("timestamp", ascending=False).copy()
    history["Rated At"] = pd.to_datetime(history["timestamp"], unit="s").dt.strftime(
        "%Y-%m-%d"
    )
    history = history.rename(
        columns={
            "movie_id": "Movie ID",
            "title": "Movie Title",
            "release_year": "Release Year",
            "rating": "User Rating",
        }
    )
    return history[
        ["Movie ID", "Movie Title", "Release Year", "User Rating", "Rated At"]
    ]


def user_genre_preferences(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    genres = display_genre_columns(movies)
    if not genres:
        return pd.DataFrame()

    history = ratings.loc[ratings["user_id"] == user_id].merge(
        movies[["movie_id", *genres]],
        on="movie_id",
        how="left",
    )
    rows = []
    for genre in genres:
        genre_rows = history.loc[history[genre] == 1]
        if genre_rows.empty:
            continue
        rows.append(
            {
                "Genre": genre,
                "Rated Movies": int(len(genre_rows)),
                "Average Rating": round(float(genre_rows["rating"].mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Average Rating", "Rated Movies"],
        ascending=False,
    )


def inject_theme() -> None:
    render_html_block(
        """
        <style>
        :root {
            --midnight: #14181c;
            --canvas: #202830;
            --shadow: #2c3440;
            --ghost: #586370;
            --steel: #667788;
            --cloud: #778899;
            --ash: #99aabb;
            --porcelain: #ddeeff;
            --white: #ffffff;
            --ocean: #445566;
            --green: #00ac1c;
            --vivid: #00e054;
            --gold: #ff9933;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(68, 85, 102, 0.28), transparent 32rem),
                linear-gradient(180deg, #14181c 0%, #101418 100%);
            color: var(--ash);
            font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif;
        }

        .block-container {
            max-width: 100% !important;
            padding-top: 0.8rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            color: var(--white) !important;
            letter-spacing: 0;
        }

        p, li, label, .stMarkdown, .stCaption, [data-testid="stMetricLabel"] {
            color: var(--ash) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(20, 24, 28, 0);
        }

        .movie-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--white);
            font-size: 1.45rem;
            font-weight: 800;
        }

        .brand-dots {
            display: inline-flex;
            gap: 0.22rem;
        }

        .brand-dots span {
            display: block;
            width: 0.82rem;
            height: 0.82rem;
            border-radius: 999px;
        }

        .dot-orange { background: #ff9933; }
        .dot-green { background: #00e054; }
        .dot-blue { background: #40bcf4; }

        .nav-links {
            display: flex;
            gap: 1rem;
            color: var(--cloud);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            min-height: 360px;
            padding: 2rem;
            background:
                radial-gradient(circle at 74% 28%, rgba(255, 153, 51, 0.22), transparent 13rem),
                linear-gradient(135deg, #202830 0%, #14181c 58%, #0f1216 100%);
            background-size: cover;
            background-position: center;
            box-shadow: rgba(0, 0, 0, 0.32) 0 18px 50px;
            border: 1px solid rgba(221, 238, 255, 0.06);
        }

        .hero-content {
            position: relative;
            z-index: 1;
            max-width: 620px;
            margin: 4.8rem auto 0;
            text-align: center;
        }

        .hero h1 {
            font-family: Georgia, "Noto Serif SC", serif;
            font-size: 2.45rem;
            line-height: 1.18;
            margin: 0 0 1rem;
        }

        .hero p {
            font-size: 1rem;
            line-height: 1.75;
            margin: 0 auto 1.4rem;
            color: var(--ash) !important;
        }

        .pill-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .pill {
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
            background: rgba(68, 85, 102, 0.72);
            color: var(--star-dust, #c8d4e0);
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid rgba(221, 238, 255, 0.08);
        }

        .section-title {
            color: var(--white);
            font-size: 1.1rem;
            font-weight: 800;
            margin: 1.5rem 0 0.7rem;
        }

        .panel {
            background: rgba(32, 40, 48, 0.82);
            border: 1px solid rgba(221, 238, 255, 0.07);
            border-radius: 8px;
            padding: 1.1rem;
            box-shadow: rgba(0, 0, 0, 0.25) 0 1px 5px 0;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 1.4rem;
        }

        .stat-card {
            background: rgba(32, 40, 48, 0.78);
            border: 1px solid rgba(221, 238, 255, 0.06);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        .stat-card .value {
            color: var(--white);
            font-size: 1.45rem;
            font-weight: 800;
        }

        .stat-card .label {
            color: var(--cloud);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .admin-subtitle {
            color: var(--cloud);
            font-size: 0.85rem;
            margin: -0.4rem 0 0.9rem;
        }

        .admin-table-card {
            background: rgba(18, 28, 38, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            padding: 1.1rem;
            box-shadow: rgba(0, 0, 0, 0.35) 0 18px 50px;
            overflow-x: auto;
            margin-bottom: 1.2rem;
        }

        .admin-table {
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            font-size: 0.86rem;
        }

        .admin-table th {
            background: rgba(0, 214, 107, 0.12);
            color: #00e676;
            font-weight: 800;
            padding: 0.7rem 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            text-align: left;
            white-space: nowrap;
        }

        .admin-table td {
            padding: 0.62rem 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            vertical-align: middle;
            white-space: nowrap;
        }

        .admin-table td.title-cell {
            max-width: 320px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .admin-table tbody tr:hover td {
            background: rgba(0, 214, 107, 0.08);
        }

        .admin-table .empty-row td {
            text-align: center;
            color: var(--cloud);
            padding: 1.2rem;
        }

        .admin-rating-badge {
            font-weight: 800;
        }

        .admin-rating-badge.high { color: #00e676; }
        .admin-rating-badge.mid { color: #ffd166; }
        .admin-rating-badge.low { color: #94a3b8; }

        .admin-badge {
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }

        .admin-badge.tier-high { background: rgba(0, 224, 84, 0.16); color: #00e676; border: 1px solid rgba(0, 224, 84, 0.45); }
        .admin-badge.tier-active { background: rgba(64, 188, 244, 0.14); color: #40bcf4; border: 1px solid rgba(64, 188, 244, 0.4); }
        .admin-badge.tier-normal { background: rgba(255, 255, 255, 0.08); color: #dbeafe; border: 1px solid rgba(255, 255, 255, 0.14); }
        .admin-badge.tier-low { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }

        /* System statistics dashboard ------------------------------------- */

        .sys-stat-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.6rem 0 1.6rem;
        }

        .sys-stat-card {
            background: linear-gradient(160deg, rgba(0, 230, 118, 0.10), rgba(18, 28, 38, 0.92));
            border: 1px solid rgba(0, 230, 118, 0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: rgba(0, 0, 0, 0.30) 0 14px 36px;
            position: relative;
            overflow: hidden;
        }

        .sys-stat-card::before {
            content: "";
            position: absolute;
            top: -40%;
            right: -30%;
            width: 90px;
            height: 90px;
            background: radial-gradient(circle, rgba(0, 230, 118, 0.30), transparent 70%);
            border-radius: 50%;
        }

        .sys-stat-card .icon {
            font-size: 1.3rem;
            margin-bottom: 0.35rem;
            opacity: 0.9;
        }

        .sys-stat-card .value {
            color: var(--white);
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0.01em;
        }

        .sys-stat-card .label {
            color: var(--cloud);
            font-size: 0.78rem;
            margin-top: 0.3rem;
            font-weight: 600;
        }

        @media (max-width: 1200px) {
            .sys-stat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }

        @media (max-width: 700px) {
            .sys-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        .chart-card-title {
            color: var(--white);
            font-size: 0.98rem;
            font-weight: 800;
            margin: 0.4rem 0 0.1rem;
        }

        .chart-card-subtitle {
            color: var(--cloud);
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stPlotlyChart"] {
            background: rgba(18, 28, 38, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            padding: 0.6rem 0.8rem;
            box-shadow: rgba(0, 0, 0, 0.35) 0 18px 50px;
            margin-bottom: 1.2rem;
        }

        .hot-movie-grid {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1.6rem;
        }

        .hot-movie-card {
            flex: 1 1 18%;
            min-width: 160px;
            height: 340px;
            background: rgba(18, 28, 38, 0.92);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
        }

        .hot-movie-rank {
            display: inline-block;
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #06281a;
            font-weight: 900;
            font-size: 0.75rem;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
            align-self: flex-start;
        }

        .hot-movie-poster-wrap {
            height: 210px;
            flex: 0 0 210px;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.5rem;
            background: var(--poster-bg, linear-gradient(135deg, #35485a, #101418));
            border: 1px solid rgba(221, 238, 255, 0.12);
        }

        .hot-movie-poster-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .hot-movie-title {
            height: 2.6em;
            line-height: 1.3;
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--white);
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .hot-movie-meta {
            margin-top: auto;
            font-size: 0.78rem;
            color: var(--cloud);
        }

        .hot-movie-meta .rating { color: #ffd166; font-weight: 800; }

        .user-table-card,
        .ranking-table-card {
            background: rgba(18, 28, 38, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            padding: 1.1rem;
            box-shadow: rgba(0, 0, 0, 0.35) 0 18px 50px;
            overflow-x: auto;
            margin-bottom: 1.2rem;
        }

        .user-dark-table,
        .ranking-table {
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            font-size: 0.86rem;
        }

        .user-dark-table th,
        .ranking-table th {
            background: rgba(0, 214, 107, 0.12);
            color: #00e676;
            font-weight: 800;
            padding: 0.7rem 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            text-align: left;
            white-space: nowrap;
        }

        .user-dark-table td,
        .ranking-table td {
            padding: 0.62rem 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            vertical-align: middle;
            white-space: nowrap;
        }

        .user-dark-table td.title-cell,
        .ranking-table td.title-cell {
            max-width: 320px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .user-dark-table tbody tr:hover td,
        .ranking-table tbody tr:hover td {
            background: rgba(0, 214, 107, 0.08);
        }

        .user-dark-table .empty-row td,
        .ranking-table .empty-row td {
            text-align: center;
            color: var(--cloud);
            padding: 1.2rem;
        }

        .rating-badge {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 193, 7, 0.12);
            color: #ffc107;
            font-weight: 800;
            white-space: nowrap;
        }

        .rank-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 64px;
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            background: rgba(0, 230, 118, 0.14);
            color: #00e676;
            font-weight: 900;
            font-size: 0.78rem;
        }

        .rank-badge.top1 {
            background: rgba(255, 193, 7, 0.18);
            color: #ffc107;
        }

        .rank-badge.top2,
        .rank-badge.top3 {
            background: rgba(0, 230, 118, 0.20);
            color: #00e676;
        }

        .recommend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .film-card {
            min-height: 222px;
        }

        .poster {
            position: relative;
            min-height: 150px;
            border-radius: 4px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(0, 0, 0, 0.36)),
                var(--poster-bg, linear-gradient(135deg, #35485a, #101418));
            border: 1px solid rgba(221, 238, 255, 0.12);
            box-shadow: rgba(0, 0, 0, 0.35) 0 8px 22px;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }

        .poster-img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: 0;
            opacity: 0.55;
        }

        .poster-rank {
            position: relative;
            z-index: 1;
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.75rem;
            font-weight: 800;
        }

        .poster-title {
            position: relative;
            z-index: 1;
            color: var(--white);
            font-family: Georgia, "Noto Serif SC", serif;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
            text-shadow: rgba(0, 0, 0, 0.55) 0 2px 8px;
        }

        .poster-genre {
            position: relative;
            z-index: 1;
            color: rgba(221, 238, 255, 0.8);
            font-size: 0.7rem;
            font-weight: 700;
            line-height: 1.25;
        }

        .detail-poster {
            min-height: 320px;
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(0, 0, 0, 0.36)),
                var(--poster-bg, linear-gradient(135deg, #35485a, #101418));
            border: 1px solid rgba(221, 238, 255, 0.12);
            box-shadow: rgba(0, 0, 0, 0.35) 0 8px 22px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .detail-poster img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .detail-poster .poster-fallback {
            color: rgba(221, 238, 255, 0.6);
            font-size: 0.85rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem;
        }

        .movie-grid-card {
            background: rgba(32, 40, 48, 0.45);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(221, 238, 255, 0.10);
            border-radius: 12px;
            padding: 0.6rem;
            height: 560px;
            display: flex;
            flex-direction: column;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }

        .movie-grid-card:hover {
            transform: translateY(-6px);
            box-shadow: rgba(0, 0, 0, 0.45) 0 18px 36px;
            border-color: rgba(0, 224, 84, 0.5);
        }

        .movie-grid-poster {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.6rem;
            height: 380px;
            flex: 0 0 380px;
            background: var(--poster-bg, linear-gradient(135deg, #35485a, #101418));
            border: 1px solid rgba(221, 238, 255, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .movie-grid-poster .poster-bg-blur {
            display: none;
        }

        .movie-grid-poster .poster-main {
            position: absolute;
            inset: 0;
            z-index: 2;
            height: 100%;
            width: 100%;
            object-fit: cover;
            object-position: center;
            display: block;
            border-radius: 8px;
        }

        .glass-poster-fallback {
            position: relative;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 0.35rem;
            padding: 0.9rem;
        }

        .movie-grid-title {
            color: var(--white);
            font-weight: 800;
            font-size: 0.95rem;
            line-height: 1.3;
            height: 2.6em;
            margin-top: 0.1rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .movie-grid-original {
            color: var(--cloud);
            font-size: 0.75rem;
            height: 1.3em;
            line-height: 1.3em;
            margin-top: 0.1rem;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }

        .movie-grid-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.4rem;
            font-size: 0.8rem;
        }

        .movie-grid-meta .rating { color: var(--gold); font-weight: 800; }
        .movie-grid-meta .year { color: var(--cloud); }

        .movie-grid-genres {
            color: rgba(221, 238, 255, 0.75);
            font-size: 0.72rem;
            margin-top: 0.3rem;
            line-height: 1.4;
            height: 2.8em;
            overflow: hidden;
        }

        /* "我的评分" cards: poster + score + review text */
        .rating-grid-card {
            background: rgba(32, 40, 48, 0.45);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(221, 238, 255, 0.10);
            border-radius: 12px;
            padding: 0.6rem;
            margin-bottom: 0.6rem;
            display: flex;
            flex-direction: column;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }

        .rating-grid-card:hover {
            transform: translateY(-6px);
            box-shadow: rgba(0, 0, 0, 0.45) 0 18px 36px;
            border-color: rgba(0, 224, 84, 0.5);
        }

        .rating-grid-poster {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.6rem;
            height: 300px;
            flex: 0 0 300px;
            background: var(--poster-bg, linear-gradient(135deg, #35485a, #101418));
            border: 1px solid rgba(221, 238, 255, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .rating-grid-poster img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
            display: block;
            border-radius: 8px;
        }

        .rating-grid-title {
            color: var(--white);
            font-weight: 800;
            font-size: 0.95rem;
            line-height: 1.3;
            height: 2.6em;
            margin-top: 0.1rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .rating-grid-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.3rem;
            font-size: 0.8rem;
        }

        .rating-grid-meta .rating { color: var(--gold); font-weight: 800; }
        .rating-grid-date { color: var(--cloud); font-size: 0.72rem; }

        .rating-grid-review {
            color: rgba(221, 238, 255, 0.85);
            font-size: 0.78rem;
            line-height: 1.5;
            margin-top: 0.4rem;
            height: 4.5em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
        }

        .rating-grid-review.empty {
            color: var(--cloud);
            font-style: italic;
        }

        .stButton > button[kind="primary"] {
            background: rgba(0, 214, 107, 0.10);
            border: 1px solid rgba(0, 214, 107, 0.55);
            color: #00e676 !important;
            box-shadow: none;
        }

        .stButton > button[kind="primary"] svg {
            fill: #00e676 !important;
            color: #00e676 !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: rgba(0, 214, 107, 0.22);
            color: #00e676 !important;
            box-shadow: none;
            transform: translateY(-2px);
        }

        /* ---------------------------------------------------------------
           Welcome / landing page — fullscreen cinematic hero background.
           The poster image + dark left-side gradient are combined into one
           `background-image` so the image fills the entire visible page
           area below the header, edge to edge; all text, stats, tags and
           the two entrance links/buttons are rendered as plain HTML inside
           this same fullscreen block. `.landing-hero-marker` lets the
           `.block-container` rule below strip the default Streamlit
           padding/max-width only on this page.
           --------------------------------------------------------------- */
        .block-container:has(.landing-hero-marker) {
            max-width: 100% !important;
            padding: 0 !important;
        }

        .landing-hero-marker {
            height: 0;
        }

        .landing-hero-banner {
            position: relative;
            width: 100vw;
            min-height: calc(100vh - 72px);
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
            margin-top: 0;
            margin-bottom: 0;
            overflow: hidden;
            border-radius: 0;
            background-color: #0f1216;
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
        }

        .landing-hero-content {
            position: relative;
            z-index: 2;
            width: min(620px, 42vw);
            padding-left: 8vw;
            padding-top: 12vh;
        }

        .landing-hero-content .landing-brand {
            font-family: Georgia, "Noto Serif SC", serif;
            font-size: 4.6rem;
            font-weight: 900;
            line-height: 1.05;
            margin: 0 0 0.4rem;
            letter-spacing: -0.02em;
        }

        .landing-hero-content .landing-brand .brand-film { color: var(--white); }
        .landing-hero-content .landing-brand .brand-trace { color: var(--vivid); }

        .landing-hero-content .landing-subtitle {
            font-size: 1.85rem;
            font-weight: 800;
            margin-top: 1.1rem;
            margin-bottom: 1rem;
            line-height: 1.25;
            color: var(--white);
        }

        .landing-hero-content .landing-description {
            color: rgba(225, 238, 255, 0.9);
            font-size: 1.05rem;
            line-height: 1.85;
            margin-bottom: 0;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.7);
        }

        .landing-stats-row {
            display: flex;
            gap: 16px;
            margin-top: 30px;
            margin-bottom: 0;
            flex-wrap: wrap;
        }

        .landing-stat {
            min-width: 130px;
            padding: 18px 20px;
            border-radius: 16px;
            background: rgba(8, 18, 32, 0.68);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .landing-stat .value {
            color: var(--vivid);
            font-size: 1.5rem;
            font-weight: 900;
            line-height: 1.3;
        }

        .landing-stat .label {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }

        .landing-badge-row {
            display: flex;
            gap: 0.7rem;
            margin-top: 18px;
            flex-wrap: wrap;
        }

        .landing-badge {
            background: rgba(0, 224, 84, 0.10);
            border: 1px solid rgba(0, 224, 84, 0.55);
            color: var(--vivid);
            border-radius: 999px;
            padding: 0.5rem 1.25rem;
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }

        .landing-hero-actions {
            display: flex;
            gap: 20px;
            margin-top: 34px;
            flex-wrap: wrap;
        }

        .landing-hero-actions .landing-hero-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 230px;
            height: 58px;
            border-radius: 18px;
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #000000 !important;
            font-weight: 900;
            font-size: 1rem;
            text-decoration: none !important;
            box-shadow: rgba(0, 230, 118, 0.32) 0 18px 42px;
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }

        .landing-hero-actions .landing-hero-btn:hover {
            transform: translateY(-2px);
            box-shadow: rgba(0, 230, 118, 0.42) 0 24px 54px;
            color: #000000 !important;
        }

        .onboarding-genre-intro {
            margin: 0.75rem 0 0.25rem 0;
        }

        .onboarding-genre-title {
            color: var(--white);
            font-weight: 700;
            font-size: 1rem;
        }

        .onboarding-genre-sub {
            color: var(--cloud);
            font-size: 0.8rem;
            margin-top: 0.15rem;
        }

        /* ---------------------------------------------------------------
           Login / register page — centered glass auth card
           --------------------------------------------------------------- */
        .auth-page-spacer {
            height: 8px;
        }

        .auth-header {
            margin-bottom: 1rem;
        }

        .auth-title {
            color: var(--white);
            font-size: 1.7rem;
            font-weight: 900;
            letter-spacing: 0.01em;
        }

        .auth-subtitle {
            color: var(--cloud);
            font-size: 0.92rem;
            margin-top: 0.35rem;
            line-height: 1.55;
        }

        .auth-hint {
            color: var(--cloud);
            font-size: 0.82rem;
            text-align: center;
            margin-top: 1rem;
        }

        .auth-genre-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 224, 84, 0.18);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin: 0.7rem 0 0.4rem;
        }

        .auth-genre-title {
            color: var(--white);
            font-weight: 800;
            font-size: 0.92rem;
        }

        .auth-genre-sub {
            color: var(--cloud);
            font-size: 0.78rem;
            margin-top: 0.25rem;
            line-height: 1.5;
        }

        /* Login/register form rendered via st.form, marked with .auth-marker so
           this styling does not leak into other forms (admin / password change).
           The auth tab/form is centered both horizontally and vertically inside
           the main content area. */
        .block-container:has(.auth-marker) {
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: calc(100vh - 110px);
            position: relative;
        }

        div[data-testid="stForm"]:has(.auth-marker) {
            width: min(70vw, 980px);
            max-width: 980px;
            min-width: 680px;
            margin: 0 auto;
            padding: 48px 64px;
            border-radius: 28px;
            background: rgba(18, 28, 38, 0.90);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-top: 3px solid rgba(0, 230, 118, 0.55);
            box-shadow: rgba(0, 0, 0, 0.45) 0 24px 70px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }

        div[data-testid="stForm"]:has(.auth-marker-register) {
            width: min(70vw, 1040px);
            max-width: 1040px;
            min-width: 720px;
            padding: 48px 68px;
        }

        div[data-testid="stForm"]:has(.auth-marker) .stTextInput,
        div[data-testid="stForm"]:has(.auth-marker) .stMultiSelect {
            width: 100%;
        }

        div[data-testid="stForm"]:has(.auth-marker) input,
        div[data-testid="stForm"]:has(.auth-marker) textarea {
            height: 46px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.30);
            color: var(--white);
            font-size: 0.95rem;
            width: 100%;
        }

        div[data-testid="stForm"]:has(.auth-marker) label p {
            color: var(--ash);
            font-weight: 700;
            font-size: 0.85rem;
        }

        div[data-testid="stForm"]:has(.auth-marker) [data-testid="stFormSubmitButton"] button {
            width: 100%;
            height: 58px;
            border-radius: 16px;
            font-size: 1rem;
        }

        @media (max-width: 1200px) {
            div[data-testid="stForm"]:has(.auth-marker) {
                width: min(88vw, 900px);
                min-width: unset;
            }
        }

        @media (max-width: 768px) {
            div[data-testid="stForm"]:has(.auth-marker) {
                width: 94vw;
                padding: 34px 28px;
            }
        }

        div[data-baseweb="select"] [data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(0, 224, 84, 0.28), rgba(0, 224, 84, 0.12)) !important;
            border: 1px solid rgba(0, 224, 84, 0.5) !important;
            border-radius: 999px !important;
            color: #000000 !important;
            font-weight: 700;
        }

        div[data-baseweb="select"] [data-baseweb="tag"] svg {
            fill: #000000 !important;
            color: #000000 !important;
        }

        .capability-card {
            background: var(--surface, rgba(255, 255, 255, 0.04));
            border: 1px solid rgba(221, 238, 255, 0.12);
            border-radius: 8px;
            padding: 1rem;
            height: 100%;
        }

        .capability-card h4 {
            margin: 0 0 0.4rem 0;
            color: var(--white);
        }

        .capability-card p {
            margin: 0;
            color: rgba(221, 238, 255, 0.8);
            font-size: 0.85rem;
            line-height: 1.5;
        }

        .film-title {
            color: var(--porcelain);
            font-size: 0.9rem;
            font-weight: 800;
            line-height: 1.35;
            margin-top: 0.55rem;
        }

        .film-meta {
            color: var(--cloud);
            font-size: 0.78rem;
            line-height: 1.55;
            margin-top: 0.18rem;
        }

        .stars {
            color: var(--gold);
            letter-spacing: 1px;
        }

        div[data-testid="stTabs"] button {
            color: var(--ash);
            font-weight: 800;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--white);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--vivid);
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #000000 !important;
            border: 0;
            border-radius: 16px;
            padding: 0.95rem 1.2rem;
            min-height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            text-shadow: none;
            box-shadow: rgba(0, 200, 83, 0.30) 0 10px 28px;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        }

        .stButton > button p,
        .stButton > button span,
        .stButton > button div,
        [data-testid="stFormSubmitButton"] button p,
        [data-testid="stFormSubmitButton"] button span,
        [data-testid="stFormSubmitButton"] button div {
            color: inherit !important;
            font-weight: inherit;
        }

        .stButton > button svg,
        [data-testid="stFormSubmitButton"] button svg {
            fill: #000000 !important;
            color: #000000 !important;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #000000 !important;
            border: 0;
            transform: translateY(-2px);
            box-shadow: rgba(0, 230, 118, 0.42) 0 14px 36px;
        }

        /* ---------------------------------------------------------------
           Sidebar redesign — dark theme to match the main content area
           --------------------------------------------------------------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0d1424 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.4rem 1.1rem 2rem;
            min-width: 240px;
            max-width: 260px;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.06);
            margin: 0.9rem 0;
        }

        /* Brand header */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.25rem;
            font-weight: 900;
            color: var(--white);
            letter-spacing: 0.01em;
            margin: 0 0 1rem 0.1rem;
        }

        .sidebar-brand .sidebar-brand-icon {
            font-size: 1.35rem;
            line-height: 1;
        }

        .sidebar-brand .sidebar-brand-trace {
            color: var(--vivid);
        }

        .sidebar-subtitle {
            color: var(--cloud);
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin: -0.5rem 0 1rem 0.15rem;
        }

        /* User info card */
        .sidebar-user-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .sidebar-user-card .sidebar-user-name {
            color: var(--white);
            font-weight: 800;
            font-size: 0.92rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .sidebar-user-card .sidebar-user-status {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--vivid);
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 0.3rem;
            letter-spacing: 0.02em;
        }

        .sidebar-user-card .sidebar-user-status::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: var(--vivid);
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.8);
            display: inline-block;
        }

        /* Ghost-style action buttons (back to home / logout) */
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.10);
            color: var(--ash) !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            min-height: 38px;
            padding: 0.4rem 0.5rem;
            border-radius: 10px;
            box-shadow: none;
            text-shadow: none;
            width: 100%;
        }

        [data-testid="stSidebar"] .stButton > button svg {
            fill: var(--ash) !important;
            color: var(--ash) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(34, 197, 94, 0.45);
            color: var(--white) !important;
            transform: none;
            box-shadow: none;
        }

        [data-testid="stSidebar"] .stButton > button:hover svg {
            fill: var(--white) !important;
            color: var(--white) !important;
        }

        /* Section labels above each nav group */
        .sidebar-section-label {
            color: rgba(148, 163, 184, 0.75);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 1rem 0.2rem 0.35rem;
        }

        /* Navigation menu — restyle st.radio as a modern row-based menu */
        [data-testid="stSidebar"] div[data-testid="stRadio"] {
            margin-bottom: 0;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.15rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            width: 100%;
            margin: 0;
            padding: 0.5rem 0.65rem;
            border-radius: 10px;
            cursor: pointer;
            color: var(--ash);
            font-size: 0.92rem;
            font-weight: 600;
            transition: background 0.15s ease, color 0.15s ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--porcelain);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(34, 197, 94, 0.15);
            color: var(--white);
            box-shadow: inset 3px 0 0 0 var(--vivid);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:last-child {
            overflow: visible;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: var(--ghost) !important;
            color: var(--porcelain) !important;
            border-radius: 12px !important;
        }

        [data-testid="stMetric"] {
            background: rgba(32, 40, 48, 0.82);
            border: 1px solid rgba(221, 238, 255, 0.08);
            border-radius: 8px;
            padding: 1rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--vivid);
            font-weight: 900;
        }

        @media (max-width: 900px) {
            .landing-hero-banner { min-height: 100vh; background-position: center right; }
            .landing-hero-content {
                width: calc(100% - 48px);
                padding-left: 24px;
                padding-right: 24px;
                padding-top: 72px;
            }
            .landing-hero-content .landing-brand { font-size: 52px; }
            .landing-hero-content .landing-subtitle { font-size: 1.5rem; }
            .landing-hero-content .landing-description { font-size: 1rem; }
            .landing-stats-row, .landing-badge-row { flex-wrap: wrap; }
            .landing-hero-actions { flex-direction: column; margin-bottom: 32px; }
            .landing-hero-actions .landing-hero-btn { width: 100%; }
        }

        @media (max-width: 760px) {
            .nav-links { display: none; }
            .hero { min-height: 430px; padding: 1.2rem; }
            .hero h1 { font-size: 1.9rem; }
            .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            div[data-testid="column"] { min-width: calc(50% - 1rem) !important; }
        }

        @media (max-width: 600px) {
            .landing-hero-content .landing-brand { font-size: 2.6rem; }
            .landing-stat { width: calc(50% - 0.65rem); }
        }
        </style>
        """
    )


def render_header() -> None:
    render_html_block(
        """
        <div class="movie-nav">
            <div class="brand">
                <span class="brand-dots">
                    <span class="dot-orange"></span>
                    <span class="dot-green"></span>
                    <span class="dot-blue"></span>
                </span>
                <span>FilmTrace</span>
            </div>
            <div class="nav-links">
                <span>评分预测</span>
                <span>相似电影</span>
                <span>MovieLens 100K</span>
            </div>
        </div>
        """
    )


LANDING_HERO_IMAGE_PATH = PROJECT_ROOT / "电影照片" / "欢迎页.jpg"


@st.cache_data(show_spinner=False)
def landing_hero_image_data_uri() -> str | None:
    """Encode the landing-page hero photo as a data URI, or None if missing."""
    if not LANDING_HERO_IMAGE_PATH.is_file():
        return None
    ext = LANDING_HERO_IMAGE_PATH.suffix.lower().lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext
    encoded = base64.b64encode(LANDING_HERO_IMAGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def render_landing_page(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """FilmTrace 落地页：单一沉浸式 Hero 卡片，海报铺满整卡，文字/统计/入口按钮叠加在左侧。"""
    # The two entrance "buttons" are plain <a> links (?entry=user / ?entry=admin)
    # rendered *inside* the hero HTML, so they stay visually inside the poster.
    # Detect them here before drawing anything else.
    requested_entry = st.query_params.get("entry")
    if requested_entry in ("user", "admin"):
        st.session_state["entry"] = requested_entry
        st.query_params.clear()
        st.rerun()

    hero_uri = landing_hero_image_data_uri()
    overlay = (
        "linear-gradient(90deg, "
        "rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.72) 30%, "
        "rgba(0,0,0,0.35) 56%, rgba(0,0,0,0.08) 100%)"
    )
    if hero_uri:
        background = f"{overlay}, url('{hero_uri}')"
    else:
        background = f"{overlay}, linear-gradient(135deg, #2b1d40, #0f1216)"

    render_html_block(
        f"""
        <div class="landing-hero-marker"></div>
        <div class="landing-hero-banner" style="background-image: {background};">
            <div class="landing-hero-content">
                <div class="landing-brand">
                    <span class="brand-film">Film</span><span class="brand-trace">Trace</span>
                </div>
                <div class="landing-subtitle">{T['landing_subtitle']}</div>
                <p class="landing-description">{T['landing_description']}</p>
                <div class="landing-stats-row">
                    <div class="landing-stat">
                        <div class="value">{ratings['user_id'].nunique():,}</div>
                        <div class="label">用户</div>
                    </div>
                    <div class="landing-stat">
                        <div class="value">{movies['movie_id'].nunique():,}</div>
                        <div class="label">部电影</div>
                    </div>
                    <div class="landing-stat">
                        <div class="value">{len(ratings):,}</div>
                        <div class="label">条评分</div>
                    </div>
                </div>
                <div class="landing-badge-row">
                    <span class="landing-badge">UserCF</span>
                    <span class="landing-badge">ItemCF</span>
                    <span class="landing-badge">SVD</span>
                </div>
                <div class="landing-hero-actions">
                    <a class="landing-hero-btn" href="?entry=user" target="_self">{T['landing_user_button']}</a>
                    <a class="landing-hero-btn" href="?entry=admin" target="_self">{T['landing_admin_button']}</a>
                </div>
            </div>
        </div>
        """
    )


def assign_movielens_user_id(ratings: pd.DataFrame, account_id: int) -> int:
    """Cycle new accounts across existing MovieLens user IDs so the existing
    recommendation models (fitted on those IDs) can serve "为你推荐" results
    for brand-new accounts without retraining."""
    user_ids = sorted(ratings["user_id"].unique())
    return int(user_ids[(account_id - 1) % len(user_ids)])


def render_auth_page(ratings: pd.DataFrame) -> None:
    """登录 / 注册界面：居中玻璃拟态卡片 + 右侧介绍面板。注册成功后写入 accounts 表并自动登录。"""
    render_html_block('<div class="auth-page-spacer"></div>')
    login_tab, register_tab = st.tabs([T["auth_login_tab"], T["auth_register_tab"]])

    with login_tab:
        with st.form("login_form"):
            render_html_block('<div class="auth-marker auth-marker-login"></div>')
            render_html_block(
                """
                <div class="auth-header">
                    <div class="auth-title">欢迎回来</div>
                    <div class="auth-subtitle">登录 FilmTrace，继续探索你的个性化电影推荐。</div>
                </div>
                """
            )
            identifier = st.text_input(
                f"{T['label_username']} / {T['label_email']}", key="user_login_username_or_email"
            )
            password = st.text_input(T["label_password"], type="password", key="user_login_password")
            submitted = st.form_submit_button(T["btn_login"], use_container_width=True)
        if submitted:
            account = accounts.authenticate_account(identifier, password)
            if account is None:
                st.error(T["msg_login_failed"])
            else:
                st.session_state["account"] = account
                st.rerun()
        render_html_block('<div class="auth-hint">还没有账号？请切换到注册页面创建账号。</div>')

    with register_tab:
        with st.form("register_form"):
            render_html_block('<div class="auth-marker auth-marker-register"></div>')
            render_html_block(
                """
                <div class="auth-header">
                    <div class="auth-title">创建 FilmTrace 账号</div>
                    <div class="auth-subtitle">选择你的电影偏好，让系统为你生成更精准的推荐。</div>
                </div>
                """
            )
            username = st.text_input(T["label_username"], key="user_register_username")
            email = st.text_input(T["label_email"], key="user_register_email")
            password = st.text_input(T["label_password"], type="password", key="user_register_password")
            confirm = st.text_input(
                T["label_confirm_password"], type="password", key="user_register_confirm_password"
            )
            render_html_block(
                f"""
                <div class="auth-genre-card">
                    <div class="auth-genre-title">{T['label_onboarding_genres']}</div>
                    <div class="auth-genre-sub">选择你感兴趣的电影类型，帮助我们为你生成专属推荐。</div>
                </div>
                """
            )
            selected_genres_zh = st.multiselect(
                T["label_onboarding_genres"],
                options=ONBOARDING_GENRES,
                key="user_register_preferences",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(T["btn_register"], use_container_width=True)
        if submitted:
            if password != confirm:
                st.error(T["msg_password_mismatch"])
            elif not selected_genres_zh:
                st.error(T["msg_onboarding_genres_required"])
            else:
                try:
                    account_id = accounts.register_account(username, email, password)
                    ml_user_id = assign_movielens_user_id(ratings, account_id)
                    conn = db.get_connection()
                    try:
                        conn.execute(
                            "UPDATE accounts SET movielens_user_id = ? WHERE account_id = ?",
                            (ml_user_id, account_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                    accounts.save_preferences(account_id, genres=selected_genres_zh, seed_movie_ids=[])
                    st.success(T["msg_register_success"])
                    st.session_state["account"] = accounts.get_account(account_id)
                    st.rerun()
                except accounts.AccountError as exc:
                    st.error(str(exc))


HERO_BANNER_PATH = PROJECT_ROOT / "电影照片" / "hero_banner.png"


@st.cache_data(show_spinner=False)
def hero_background_data_uri() -> str | None:
    """Encode the local hero banner image as a data URI for use as a CSS background.

    Returns None if the file is missing, so `render_hero` can fall back to
    the plain gradient background defined in `inject_theme`.
    """
    if not HERO_BANNER_PATH.is_file():
        return None
    encoded = base64.b64encode(HERO_BANNER_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_hero(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    bg_uri = hero_background_data_uri()
    hero_style = ""
    if bg_uri:
        hero_style = (
            ' style="background-image: '
            "linear-gradient(rgba(15, 18, 22, 0.78), rgba(15, 18, 22, 0.85)), "
            f"url(&quot;{bg_uri}&quot;);\""
        )
    render_html_block(
        f"""
        <section class="hero"{hero_style}>
            <div class="hero-content">
                <h1>FilmTrace 电影推荐系统</h1>
                <p>
                    Predict Ratings. Discover Movies. Explore Recommendation Algorithms.<br>
                    基于 MovieLens 100K 数据集构建 · UserCF · ItemCF · SVD · Hybrid Recommendation
                </p>
                <div class="pill-row">
                    <span class="pill">{ratings['user_id'].nunique():,} 位用户</span>
                    <span class="pill">{movies['movie_id'].nunique():,} 部电影</span>
                    <span class="pill">{len(ratings):,} 条评分</span>
                    <span class="pill">UserCF · ItemCF · SVD · Hybrid</span>
                </div>
            </div>
        </section>
        """
    )


def style_plotly_dark(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply the FilmTrace dark/teal theme to a Plotly figure and return it."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="'Segoe UI', system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        hoverlabel=dict(bgcolor="#16222c", font_color="#e7f3ff", bordercolor="#00e676"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9fb3c8")),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, color="#9fb3c8", linecolor="rgba(255,255,255,0.12)", zeroline=False)
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#9fb3c8", zeroline=False
    )
    return fig


def render_chart_card(title: str, subtitle: str, fig: go.Figure) -> None:
    """Render a titled Plotly chart styled as a dark dashboard card."""
    st.markdown(
        f'<div class="chart-card-title">{html.escape(title)}</div>'
        f'<div class="chart-card-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_dataset_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    sparsity = 1 - len(ratings) / (
        ratings["user_id"].nunique() * movies["movie_id"].nunique()
    )
    render_html_block(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="value">{ratings['user_id'].nunique():,}</div><div class="label">用户数</div></div>
            <div class="stat-card"><div class="value">{movies['movie_id'].nunique():,}</div><div class="label">电影数</div></div>
            <div class="stat-card"><div class="value">{len(ratings):,}</div><div class="label">评分总数</div></div>
            <div class="stat-card"><div class="value">{sparsity * 100:.1f}%</div><div class="label">数据稀疏度</div></div>
        </div>
        """
    )


def render_dark_table(
    display_df: pd.DataFrame,
    table_class: str = "admin-table",
    card_class: str = "admin-table-card",
    html_columns: set[str] | None = None,
) -> None:
    """Render `display_df` as a dark HTML table inside a `card_class` card.

    Columns listed in `html_columns` are inserted as raw HTML (e.g. badges /
    rating pills built by `admin_rating_badge_html` / `activity_tier_badge_html` /
    `rating_badge_html` / `rank_badge_html`); all other cell values are
    HTML-escaped plain text. Columns named "标题"/"中文片名"/"原片名"/"电影标题"
    get a `title-cell` class so long titles are clipped with an ellipsis.
    """
    html_columns = html_columns or set()
    if display_df.empty:
        render_html_block(
            f'<div class="{card_class}"><table class="{table_class}">'
            '<tbody><tr class="empty-row"><td>暂无数据</td></tr></tbody></table></div>'
        )
        return

    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in display_df.columns)
    body_rows = []
    for _, row in display_df.iterrows():
        cells = []
        for col in display_df.columns:
            value = row[col]
            if col in html_columns:
                cells.append(f"<td>{value}</td>")
            elif col in ("标题", "中文片名", "原片名", "电影标题"):
                cells.append(f'<td class="title-cell">{html.escape(str(value))}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    render_html_block(
        f'<div class="{card_class}"><table class="{table_class}">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def render_admin_table(display_df: pd.DataFrame, html_columns: set[str] | None = None) -> None:
    """Render `display_df` as a dark `.admin-table` inside an `.admin-table-card`."""
    render_dark_table(display_df, table_class="admin-table", card_class="admin-table-card", html_columns=html_columns)


def admin_rating_badge_html(value: float) -> str:
    """Return a `.admin-rating-badge` span colored by rating tier (green/yellow/gray)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="admin-rating-badge low">暂无</span>'
    value = float(value)
    tier = "high" if value >= 4 else "mid" if value >= 3 else "low"
    return f'<span class="admin-rating-badge {tier}">⭐ {value:.2f}</span>'


def activity_tier_badge_html(rating_count: int) -> str:
    """Return an `.admin-badge` span classifying a user by rating-count activity level."""
    rating_count = int(rating_count)
    if rating_count >= 200:
        return '<span class="admin-badge tier-high">高活跃</span>'
    if rating_count >= 100:
        return '<span class="admin-badge tier-active">活跃</span>'
    if rating_count >= 20:
        return '<span class="admin-badge tier-normal">普通</span>'
    return '<span class="admin-badge tier-low">低活跃</span>'


def rating_badge_html(value: float) -> str:
    """Return a `.rating-badge` span showing a user's star rating, e.g. "⭐ 5"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="rating-badge">暂无</span>'
    value = float(value)
    text = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    return f'<span class="rating-badge">⭐ {text}</span>'


def rank_badge_html(rank: int) -> str:
    """Return a `.rank-badge` span labeled "TOP {rank}", with extra accents for the top 3."""
    rank = int(rank)
    tier = f"top{rank}" if rank <= 3 else ""
    css_class = f"rank-badge {tier}".strip()
    return f'<span class="{css_class}">TOP {rank}</span>'


def render_capability_cards() -> None:
    capabilities = [
        ("UserCF", "基于相似用户的历史评分，为目标用户推荐其“同好”喜欢的电影。"),
        ("ItemCF", "基于物品-物品相似度，查找与目标电影评分模式相近的其他电影。"),
        ("SVD", "通过矩阵分解学习用户与电影的潜在因子，预测精度最高。"),
        ("Hybrid Model", "结合协同过滤与电影元数据特征的混合推荐模型，缓解冷启动问题。"),
        ("推荐评估", "基于 RMSE / MAE / Precision@K / Recall@K / NDCG@K 等指标系统评估各算法。"),
        ("数据可视化", "评分分布、用户行为、电影类型分布等多维度可视化分析。"),
        ("冷启动分析", "分析新用户 / 新电影在评分数据不足时的推荐表现。"),
        ("稀疏度分析", "评估评分矩阵稀疏度对协同过滤效果的影响。"),
    ]
    for start in range(0, len(capabilities), 4):
        cap_cols = st.columns(4)
        for col, (name, desc) in zip(cap_cols, capabilities[start : start + 4], strict=False):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(desc)


def poster_background(rank: int) -> str:
    return poster_gradient(rank)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_poster_url(title: str, year: object = None) -> str | None:
    """Look up a TMDb poster URL, cached for the session (returns None offline)."""
    parsed_year = int(year) if isinstance(year, int | float) and not pd.isna(year) else None
    return fetch_poster_url(title, parsed_year)


def render_fallback_poster(rank: int, title_zh: str, genres_zh: str = "", year: object = None) -> None:
    """Render a clean local poster placeholder showing 中文片名/年份/中文类型.

    Used when no TMDb poster is available, so the UI never depends on an
    external API and never shows raw HTML as text.
    """
    bg = poster_background(rank)
    year_text = ""
    if year is not None and not (isinstance(year, float) and pd.isna(year)) and str(year).strip():
        year_text = f"（{year}）"
    parts = [
        '<div style="background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(0,0,0,0.46)), ',
        f'{bg}; border-radius: 6px; min-height: 150px; padding: 0.9rem; ',
        'display: flex; flex-direction: column; align-items: center; justify-content: center; ',
        'text-align: center; gap: 0.35rem;">',
        f'<div style="color: rgba(255,255,255,0.65); font-size: 0.7rem; font-weight: 800;">#{rank}</div>',
        '<div style="color: #ffffff; font-weight: 800; font-size: 1rem; line-height: 1.3;">',
        f'{html.escape(title_zh)}{html.escape(year_text)}</div>',
    ]
    if genres_zh:
        parts.append(
            f'<div style="color: rgba(221,238,255,0.85); font-size: 0.75rem;">{html.escape(genres_zh)}</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_poster(
    identifier: int,
    title: str,
    title_zh: str,
    genres_zh: str = "",
    year: object = None,
) -> None:
    """Render a movie poster: local image first, then TMDb, then a fallback card.

    Local posters in `电影照片/` are matched by Chinese display title or
    original English title, so the demo never depends on TMDb during a
    defense/presentation.
    """
    local_path = local_poster_path(title, title_zh)
    if local_path is not None:
        st.image(str(local_path), use_container_width=True)
        return
    poster_url = cached_poster_url(title, year)
    if poster_url:
        st.image(poster_url, use_container_width=True)
        return
    render_fallback_poster(identifier, title_zh, genres_zh, year)


def star_text(avg_rating: float) -> str:
    if pd.isna(avg_rating):
        return "暂无评分"
    filled = max(0, min(5, round(avg_rating)))
    return f"{filled}/5"


@st.cache_data(show_spinner=False)
def poster_data_uri(path_str: str) -> str:
    """Base64-encode a local poster image for inline embedding in raw HTML cards."""
    path = Path(path_str)
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def library_poster_html(
    identifier: int,
    title: str,
    title_zh: str,
    genres_zh: str = "",
    year: object = None,
    css_class: str = "movie-grid-poster",
) -> str:
    """Return an HTML snippet for a movie poster: local image, TMDb image, or gradient fallback.

    Used inside glassmorphism movie cards where the poster must be part of a
    single raw-HTML block (so the hover animation covers the whole card).
    `css_class` selects the wrapper class (e.g. `movie-grid-poster` or
    `hot-movie-poster-wrap`), which controls the poster's fixed size.
    """
    use_poster_stage = css_class == "movie-grid-poster"

    def poster_markup(src: str) -> str:
        alt = html.escape(title_zh)
        if use_poster_stage:
            return (
                f'<div class="{css_class}">'
                f'<img class="poster-bg-blur" src="{src}" alt="">'
                f'<img class="poster-main" src="{src}" alt="{alt}">'
                "</div>"
            )
        return f'<div class="{css_class}"><img src="{src}" alt="{alt}"></div>'

    local_path = local_poster_path(title, title_zh)
    if local_path is not None:
        uri = poster_data_uri(str(local_path))
        return poster_markup(uri)

    poster_url = cached_poster_url(title, year)
    if poster_url:
        return poster_markup(poster_url)

    bg = poster_background(identifier)
    year_text = ""
    if year is not None and not (isinstance(year, float) and pd.isna(year)) and str(year).strip():
        year_text = f"（{year}）"
    genre_html = (
        f'<div style="color: rgba(221,238,255,0.85); font-size: 0.75rem;">{html.escape(genres_zh)}</div>'
        if genres_zh
        else ""
    )
    return (
        f'<div class="{css_class}" style="background: linear-gradient(180deg, '
        f"rgba(255,255,255,0.10), rgba(0,0,0,0.46)), {bg};\">"
        '<div class="glass-poster-fallback">'
        f'<div style="color: rgba(255,255,255,0.65); font-size: 0.7rem; font-weight: 800;">#{identifier}</div>'
        '<div style="color: #ffffff; font-weight: 800; font-size: 1rem; line-height: 1.3;">'
        f"{html.escape(title_zh)}{html.escape(year_text)}</div>"
        f"{genre_html}"
        "</div></div>"
    )


@st.dialog("电影简介")
def show_movie_intro(detail: dict[str, object]) -> None:
    st.markdown(f"### {detail['title']}")
    st.write(f"在当前相似电影推荐列表中排名第 {detail['rank']} 位。")
    st.metric("相似度", f"{float(detail['similarity']):.4f}")
    intro_cols = st.columns(3)
    intro_cols[0].metric("平均评分", f"{float(detail['avg_rating']):.2f} / 5")
    intro_cols[1].metric("评分数", f"{int(detail['rating_count']):,}")
    intro_cols[2].metric("上映年份", str(detail["release_year"]))
    st.markdown(f"**类型：** {html.escape(str(detail['genres']))}")
    st.markdown(
        "推荐该电影的原因：在 MovieLens 100K 评分矩阵中，"
        "对所选电影评分较高的用户对该电影也表现出相似的评分模式。"
    )
    imdb_url = str(detail.get("imdb_url", ""))
    st.link_button(T["btn_open_imdb"], build_safe_imdb_url(str(detail["title"]), imdb_url))
    st.caption("MovieLens 100K 提供了元数据和 IMDb 链接，但不包含海报图片或剧情简介。")


def render_recommendation_cards(
    recs: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> None:
    for start in range(0, len(recs), 5):
        cols = st.columns(5)
        for col, (_, rec) in zip(cols, recs.iloc[start : start + 5].iterrows()):
            avg = (
                float(rec["Avg Rating"])
                if not pd.isna(rec["Avg Rating"])
                else float("nan")
            )
            movie_id = int(rec["Movie ID"])
            movie_row = movies.loc[movies["movie_id"] == movie_id].iloc[0]
            genres_zh = translate_genres(movie_genre_text(movie_row, movies))
            title = str(rec["Title"])
            title_zh = get_display_title(title)
            rank = int(rec["Rank"])
            with col:
                with st.container(border=True):
                    render_poster(rank, title, title_zh, genres_zh, movie_row.get("release_year"))
                    st.markdown(f"**{rank}. {title_zh}**")
                    if title_zh != title:
                        st.caption(title)
                    st.caption(genres_zh)
                    st.caption(f"相似度 {float(rec['Similarity']):.4f}　平均评分 {avg:.2f} {star_text(avg)}")
                    if st.button(T["btn_view_intro"], key=f"intro_{movie_id}_{rank}"):
                        show_movie_intro(movie_detail(movie_id, recs, movies, ratings))


def render_hot_movies_top5(catalog: pd.DataFrame, top_n: int = 5) -> None:
    """Admin home: 热门电影 Top N as a row of equal-size `.hot-movie-card` cards."""
    rows = popular_movies(catalog, "Weighted Score", top_n).reset_index(drop=True)
    if rows.empty:
        st.info("暂无数据。")
        return

    cards = []
    for idx, row in rows.iterrows():
        rank = idx + 1
        avg = float(row["Average Rating"]) if not pd.isna(row["Average Rating"]) else float("nan")
        title = str(row["Title"])
        title_zh = get_display_title(title)
        genres_zh = translate_genres(str(row["Genres"]))
        movie_id = int(row["Movie ID"])
        rating_text = f"⭐ {avg:.2f}" if not pd.isna(avg) else "暂无评分"
        cards.append(
            f"""
            <div class="hot-movie-card">
                <span class="hot-movie-rank">Top {rank}</span>
                {library_poster_html(movie_id, title, title_zh, genres_zh, row.get("Release Year"), css_class="hot-movie-poster-wrap")}
                <div class="hot-movie-title">{html.escape(title_zh)}</div>
                <div class="hot-movie-meta">
                    <span class="rating">{rating_text}</span>　评分数 {int(row['Ratings']):,}
                </div>
            </div>
            """
        )
    render_html_block(f'<div class="hot-movie-grid">{"".join(cards)}</div>')


def render_catalog_cards(catalog_rows: pd.DataFrame, rank_offset: int = 1) -> None:
    """Render movie cards (poster, title, genres, average rating) for a catalog-shaped DataFrame."""
    rows = catalog_rows.reset_index(drop=True)
    for start in range(0, len(rows), 5):
        cols = st.columns(5)
        for col, (idx, row) in zip(cols, rows.iloc[start : start + 5].iterrows(), strict=False):
            rank = idx + rank_offset
            avg = (
                float(row["Average Rating"])
                if not pd.isna(row["Average Rating"])
                else float("nan")
            )
            title = str(row["Title"])
            title_zh = get_display_title(title)
            genres_zh = translate_genres(str(row["Genres"]))
            with col:
                with st.container(border=True):
                    render_poster(rank, title, title_zh, genres_zh, row.get("Release Year"))
                    st.markdown(f"**{title_zh}**")
                    if title_zh != title:
                        st.caption(title)
                    st.caption(genres_zh)
                    st.caption(f"平均评分 {avg:.2f} {star_text(avg)}　评分数 {int(row['Ratings']):,}")


@st.dialog("电影详情")
def show_library_movie_detail(movie_id: int, catalog: pd.DataFrame, movies: pd.DataFrame) -> None:
    """详情面板：海报、中英文片名、类型、上映年份、平均评分、评分数、IMDb 链接。"""
    row = catalog.loc[catalog["Movie ID"] == movie_id].iloc[0]
    avg = float(row["Average Rating"]) if not pd.isna(row["Average Rating"]) else float("nan")
    title = str(row["Title"])
    title_zh = get_display_title(title)
    genres_zh = translate_genres(str(row["Genres"]))
    year = row.get("Release Year")

    detail_cols = st.columns([0.36, 0.64])
    with detail_cols[0]:
        render_poster(movie_id, title, title_zh, genres_zh, year)
    with detail_cols[1]:
        st.markdown(f"### {title_zh}")
        if title_zh != title:
            st.caption(f"原片名：{title}")
        st.markdown(f"**{T['label_genre']}：** {genres_zh}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("上映年份", str(year) if year and str(year).strip() else "未知")
        metric_cols[1].metric("平均评分", f"{avg:.2f} / 5" if not pd.isna(avg) else "暂无评分")
        metric_cols[2].metric("评分数", f"{int(row['Ratings']):,}")
        movie_row = movies.loc[movies["movie_id"] == movie_id].iloc[0]
        imdb_url = str(movie_row.get("imdb_url", ""))
        st.link_button(T["btn_open_imdb"], build_safe_imdb_url(title, imdb_url))


@st.dialog("评价电影")
def show_rate_movie_dialog(movie_id: int, title_zh: str, ml_user_id: int) -> None:
    """评价弹窗：给出 1-5 分评分与评论文本，提交后写入 `ratings` 表（含 review 列）。"""
    st.markdown(f"### {title_zh}")
    existing = db.get_user_rating(ml_user_id, movie_id)
    default_rating = int(existing["rating"]) if existing is not None else 5
    default_review = str(existing["review"]) if existing is not None else ""
    if existing is not None:
        st.caption("你之前已评价过这部电影，可在下方修改。")

    rating = st.slider("评分", min_value=1, max_value=5, value=default_rating, key=f"rate_score_{movie_id}")
    review = st.text_area(
        "评价 / 评论（可选）",
        value=default_review,
        key=f"rate_review_{movie_id}",
        height=120,
        placeholder="写下你对这部电影的看法……",
    )

    btn_cols = st.columns(2)
    if btn_cols[0].button("提交", type="primary", use_container_width=True, key=f"rate_submit_{movie_id}"):
        db.upsert_rating(ml_user_id, movie_id, int(rating), review.strip())
        st.success("评价已保存，可在“我的评分”页面查看。")
        st.rerun()
    if btn_cols[1].button("取消", use_container_width=True, key=f"rate_cancel_{movie_id}"):
        st.rerun()


def movie_card_html(row: pd.Series) -> str:
    """Build the `.movie-grid-card` HTML for one catalog row (poster, 中文/原片名, 评分, 类型, 年份)."""
    movie_id = int(row["Movie ID"])
    avg = float(row["Average Rating"]) if not pd.isna(row["Average Rating"]) else float("nan")
    title = str(row["Title"])
    title_zh = get_display_title(title)
    genres_zh = translate_genres(str(row["Genres"]))
    year = row.get("Release Year")
    rating_text = f"⭐ {avg:.2f}" if not pd.isna(avg) else "暂无评分"
    year_text = str(year) if year and str(year).strip() else "未知"
    original_html = f'<div class="movie-grid-original">{html.escape(title)}</div>' if title_zh != title else '<div class="movie-grid-original"></div>'
    return f"""
        <div class="movie-grid-card">
            {library_poster_html(movie_id, title, title_zh, genres_zh, year)}
            <div class="movie-grid-title">{html.escape(title_zh)}</div>
            {original_html}
            <div class="movie-grid-meta">
                <span class="rating">{rating_text}</span>
                <span class="year">{html.escape(year_text)}</span>
            </div>
            <div class="movie-grid-genres">{html.escape(genres_zh)}</div>
        </div>
        """


def render_library_card_grid(
    catalog_rows: pd.DataFrame, movies: pd.DataFrame, columns: int = 4, ml_user_id: int | None = None
) -> None:
    """Render movies as a responsive glassmorphism card grid (poster, 中文/原片名, 评分, 类型, 年份).

    Clicking "查看详情" opens a modal detail panel via `show_library_movie_detail`,
    "收藏" toggles the favorite, and "评价" opens a rating/review dialog whose
    submission is written to the `ratings` table and shown on "我的评分".
    """
    rows = catalog_rows.reset_index(drop=True)
    if rows.empty:
        st.info("未找到匹配电影，请尝试其他关键词。")
        return

    account = st.session_state.get("account")
    account_id = int(account["account_id"]) if account is not None else None

    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (_, row) in zip(cols, rows.iloc[start : start + columns].iterrows(), strict=False):
            movie_id = int(row["Movie ID"])
            title_zh = get_display_title(str(row["Title"]))
            with col:
                render_html_block(movie_card_html(row))
                action_cols = st.columns(2)
                if action_cols[0].button("查看详情", key=f"lib_card_{movie_id}", use_container_width=True):
                    show_library_movie_detail(movie_id, catalog_rows, movies)
                is_fav = account_id is not None and accounts.is_favorite(account_id, movie_id)
                fav_label = "♥ 已收藏" if is_fav else "♡ 收藏"
                if action_cols[1].button(fav_label, key=f"lib_fav_{movie_id}", use_container_width=True, type="primary"):
                    if account_id is None:
                        st.warning("请先登录后再收藏")
                    elif is_fav:
                        accounts.remove_favorite(account_id, movie_id)
                        st.rerun()
                    else:
                        accounts.add_favorite(account_id, movie_id)
                        st.success("收藏成功")
                        st.rerun()
                if st.button("⭐ 评价", key=f"lib_rate_{movie_id}", use_container_width=True):
                    if ml_user_id is None:
                        st.warning("请先登录后再评价")
                    else:
                        show_rate_movie_dialog(movie_id, title_zh, ml_user_id)


def render_movie_detail_page(catalog: pd.DataFrame, movies: pd.DataFrame) -> None:
    """电影详情页：poster, title, genres, release year, average rating, rating count."""
    title_opts = build_title_options(movies)
    selected_label = st.selectbox(
        T["label_title"],
        options=title_opts["label"].tolist(),
        index=0,
        key="detail_movie_select",
    )
    selected_id = int(title_opts.loc[title_opts["label"] == selected_label, "movie_id"].iloc[0])
    row = catalog.loc[catalog["Movie ID"] == selected_id].iloc[0]
    avg = float(row["Average Rating"]) if not pd.isna(row["Average Rating"]) else float("nan")
    title = str(row["Title"])
    title_zh = get_display_title(title)
    genres_zh = translate_genres(str(row["Genres"]))
    detail_cols = st.columns([0.3, 0.7])
    with detail_cols[0]:
        render_poster(selected_id, title, title_zh, genres_zh, row.get("Release Year"))
    with detail_cols[1]:
        st.markdown(f"### {title_zh}")
        if title_zh != title:
            st.caption(f"原片名：{title}")
        st.markdown(f"**{T['label_genre']}：** {genres_zh}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("上映年份", str(row.get("Release Year", "未知")))
        metric_cols[1].metric("平均评分", f"{avg:.2f} / 5  {star_text(avg)}")
        metric_cols[2].metric("评分数", f"{int(row['Ratings']):,}")
        movie_row = movies.loc[movies["movie_id"] == selected_id].iloc[0]
        imdb_url = str(movie_row.get("imdb_url", ""))
        st.link_button(T["btn_open_imdb"], build_safe_imdb_url(title, imdb_url))


def render_user_summary(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    preferred_genres_zh: list[str] | None = None,
) -> None:
    history_df = user_history_table(ratings, movies, user_id)
    preferred_text = "、".join(preferred_genres_zh) if preferred_genres_zh else "暂无偏好"
    liked_count = int((history_df["User Rating"] >= 4).sum())
    cols = st.columns(5)
    cols[0].metric("已评分电影数", f"{len(history_df):,}")
    cols[1].metric("平均评分", f"{history_df['User Rating'].mean():.2f}")
    cols[2].metric("偏好类型", preferred_text)
    cols[3].metric("高分电影数", f"{liked_count:,}")
    cols[4].metric("最后评分日期（历史数据）", str(history_df["Rated At"].iloc[0]))
    st.caption(
        "MovieLens 100K 数据集评分时间主要集中在 1997-1998 年，因此日期为历史数据，不代表当前时间。"
    )


def render_personalized_cards(recs: pd.DataFrame) -> None:
    if recs.empty:
        st.warning(T["msg_no_candidates"])
        return
    for start in range(0, len(recs), 5):
        cols = st.columns(5)
        for col, (_, rec) in zip(cols, recs.iloc[start : start + 5].iterrows(), strict=False):
            rank = int(rec["Rank"])
            title = str(rec["Title"])
            title_zh = get_display_title(title)
            genres_zh = translate_genres(str(rec["Genres"]))
            year = rec.get("Release Year")
            with col:
                with st.container(border=True):
                    render_poster(rank, title, title_zh, genres_zh, year)
                    st.markdown(f"**{rank}. {title_zh}**")
                    if title_zh != title:
                        st.caption(title)
                    st.caption(genres_zh)
                    st.caption(
                        f"预测评分 {float(rec['Recommendation Score']):.2f} / 5　"
                        f"平均评分 {float(rec['Average Rating']):.2f} / 5　"
                        f"评分人数 {int(rec['Ratings']):,}"
                    )
                    st.caption(f"置信度 {float(rec['Confidence']):.2f}")
                    st.caption(f"推荐理由：{rec['Reason']}")


def render_home_page(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    catalog: pd.DataFrame,
    user_id: int,
) -> None:
    render_hero(ratings, movies)

    st.markdown('<div class="section-title">数据集概览</div>', unsafe_allow_html=True)
    render_dataset_stats(ratings, movies)

    st.markdown('<div class="section-title">热门电影</div>', unsafe_allow_html=True)
    st.caption("MovieLens 100K 中综合评分最高、最受欢迎的电影，海报优先使用本地图库。")
    render_catalog_cards(popular_movies(catalog, "Weighted Score", 10))

    st.markdown('<div class="section-title">推荐系统介绍</div>', unsafe_allow_html=True)
    render_capability_cards()


def render_for_you_page(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    catalog: pd.DataFrame,
    user_id: int,
) -> None:
    account = st.session_state.get("account")
    preferred_genres_zh: list[str] = []
    if account is not None:
        prefs = accounts.get_preferences(account["account_id"])
        if prefs:
            preferred_genres_zh = prefs.get("genres", [])

    render_user_summary(ratings, movies, user_id, preferred_genres_zh)
    st.markdown('<div class="section-title">为你推荐</div>', unsafe_allow_html=True)
    st.caption(f"基于用户 #{user_id} 的历史评分，系统已自动生成以下个性化推荐结果。")

    if preferred_genres_zh:
        st.markdown(f"**{T['label_user_preferred_genres']}：** {'、'.join(preferred_genres_zh)}")

    with st.expander("高级参数设置"):
        ctrl = st.columns(4)
        algorithm = ctrl[0].selectbox(
            T["label_algorithm"],
            [
                "Item-based Collaborative Filtering",
                "User-based Collaborative Filtering",
                "SVD Matrix Factorization",
            ],
        )
        top_n = ctrl[1].selectbox(T["label_recommendation_count"], [5, 10, 20], index=1)
        k = ctrl[2].slider(T["label_neighbors_k"], 5, 50, int(st.session_state.get("admin_k", 20)), 5)
        factors = ctrl[3].slider(T["label_svd_factors"], 10, 80, int(st.session_state.get("admin_factors", 50)), 10)
        metric = st.selectbox(
            T["label_similarity_method"],
            ["cosine", "pearson"],
            index=0 if st.session_state.get("admin_metric", "cosine") == "cosine" else 1,
        )
        st.button(T["btn_generate_recommendations"])
        st.caption("首次生成可能需要数秒，后续相同参数将直接读取缓存。")

    n_ratings = int((ratings["user_id"] == user_id).sum())
    elapsed = 0.0
    source = "popular"
    recs = pd.DataFrame()
    if n_ratings >= 5:
        recs, elapsed = safe_personalized_recommendations(
            ratings,
            movies,
            avg_ratings,
            rating_counts,
            user_id,
            algorithm,
            int(top_n),
            int(k),
            metric,
            int(factors),
        )
        if not recs.empty:
            source = "cf"
            recs = apply_genre_preference_boost(recs, preferred_genres_zh)
        elif preferred_genres_zh:
            recs = genre_based_recommendations(catalog, preferred_genres_zh, int(top_n))
            source = "genre" if not recs.empty else "popular"
    elif preferred_genres_zh:
        recs = genre_based_recommendations(catalog, preferred_genres_zh, int(top_n))
        source = "genre" if not recs.empty else "popular"

    if recs.empty:
        recs = popularity_recommendations(catalog, int(top_n))
        source = "popular"

    st.caption(f"推荐生成耗时：{elapsed:.2f} 秒")
    st.markdown(f"**{T['label_recommendation_source']}：** {T[f'recommendation_source_{source}']}")
    render_personalized_cards(recs)

    if source == "cf":
        st.markdown('<div class="section-title">推荐说明</div>', unsafe_allow_html=True)
        st.markdown(recommendation_explanation(algorithm))

        st.markdown('<div class="section-title">模型评估指标摘要</div>', unsafe_allow_html=True)
        st.caption("以下为各算法在按用户时间序列划分的留出测试集上的评分预测误差（数值越低越好）。")
        eval_table = evaluation_summary(ratings)
        metric_cols = st.columns(len(eval_table))
        for col, (_, row) in zip(metric_cols, eval_table.iterrows(), strict=False):
            col.metric(str(row["Algorithm"]), f"RMSE {row['RMSE']:.3f}", f"MAE {row['MAE']:.3f}")

    with st.expander("查看推荐数据"):
        st.dataframe(translate_columns(recs), use_container_width=True, hide_index=True)


def render_catalog_page(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    catalog: pd.DataFrame,
    user_id: int,
) -> None:
    catalog_tabs = st.tabs(
        [
            "浏览与搜索",
            "电影详情页",
            T["nav_rating_records"],
            "热门电影",
            T["nav_visualization"],
            T["nav_similar_movies"],
        ]
    )

    with catalog_tabs[0]:
        st.markdown('<div class="section-title">电影库</div>', unsafe_allow_html=True)
        st.caption("支持按电影ID、电影名称和类型筛选 MovieLens 电影数据。")
        st.caption(
            f"本地海报目录：{LOCAL_POSTER_DIR} / 新增海报后点击“刷新海报库”即可更新显示。"
        )
        if st.button("🔄 刷新海报库", key="refresh_poster_cache"):
            refresh_poster_cache()
            poster_data_uri.clear()
            cached_poster_url.clear()
            st.success("海报库已刷新。")
        search_cols = st.columns([0.65, 0.35])
        query = search_cols[0].text_input(f"{T['btn_search']}（按标题或电影ID）")
        genre = search_cols[1].selectbox(T["label_genre"], [T["label_all"], *genre_options_zh(movies)])

        results = search_movies(catalog, query, genre)
        if results.empty:
            st.info("未找到匹配电影，请尝试其他关键词。")
        else:
            is_default_view = not query.strip() and genre == T["label_all"]
            display_results = results.head(20) if is_default_view else results.head(80)
            if is_default_view:
                st.caption("当前显示热门高分电影 Top 20，输入关键词或选择类型可进一步筛选。")
            render_library_card_grid(display_results, movies, ml_user_id=user_id)

    with catalog_tabs[1]:
        st.markdown('<div class="section-title">电影详情页</div>', unsafe_allow_html=True)
        render_movie_detail_page(catalog, movies)

    with catalog_tabs[2]:
        st.markdown('<div class="section-title">用户评分记录</div>', unsafe_allow_html=True)
        st.caption("MovieLens 100K 数据集记录了用户的显式评分及对应的 Unix 时间戳。")

        history = user_history_table(ratings, movies, user_id)
        rating_metric_cols = st.columns(4)
        rating_metric_cols[0].metric("已评分电影数", f"{len(history):,}")
        rating_metric_cols[1].metric(
            "平均评分", f"{history['User Rating'].mean():.2f}" if not history.empty else "暂无"
        )
        rating_metric_cols[2].metric(
            "最高评分", f"{history['User Rating'].max():.0f}" if not history.empty else "暂无"
        )
        rating_metric_cols[3].metric(
            "最近评分日期", history["Rated At"].iloc[0] if not history.empty else "暂无"
        )

        display_history = pd.DataFrame(
            {
                "电影ID": history["Movie ID"],
                "电影标题": history["Movie Title"].map(get_display_title),
                "上映年份": history["Release Year"],
                "用户评分": history["User Rating"].map(rating_badge_html),
                "评分日期": history["Rated At"],
            }
        )
        render_dark_table(display_history, table_class="user-dark-table", card_class="user-table-card", html_columns={"用户评分"})

        genre_pref = user_genre_preferences(ratings, movies, user_id)
        st.markdown('<div class="section-title">偏好分析</div>', unsafe_allow_html=True)
        st.dataframe(translate_columns(genre_pref), use_container_width=True, hide_index=True)

    with catalog_tabs[3]:
        st.markdown('<div class="section-title">热门电影排行</div>', unsafe_allow_html=True)

        with st.container(border=True):
            rank_cols = st.columns(2)
            rank_by = rank_cols[0].selectbox(
                T["label_rank_by"],
                ["Weighted Score", "Average Rating", "Number of Ratings"],
            )
            pop_n = rank_cols[1].selectbox(T["label_show_top"], [10, 20], index=0)

        ranking = popular_movies(catalog, rank_by, int(pop_n)).reset_index(drop=True)
        rank_metric_cols = st.columns(4)
        rank_metric_cols[0].metric("上榜电影数", f"{len(ranking):,}")
        rank_metric_cols[1].metric(
            "最高加权评分", f"{ranking['Weighted Score'].max():.3f}" if not ranking.empty else "暂无"
        )
        rank_metric_cols[2].metric(
            "平均评分", f"{ranking['Average Rating'].mean():.2f}" if not ranking.empty else "暂无"
        )
        rank_metric_cols[3].metric(
            "最高评分数", f"{int(ranking['Ratings'].max()):,}" if not ranking.empty else "暂无"
        )

        display_ranking = pd.DataFrame(
            {
                "排名": [rank_badge_html(idx + 1) for idx in range(len(ranking))],
                "电影ID": ranking["Movie ID"],
                "标题": ranking["Title"].map(get_display_title),
                "类型": ranking["Genres"].map(translate_genres),
                "平均评分": ranking["Average Rating"].map(admin_rating_badge_html),
                "评分数": ranking["Ratings"],
                "上映年份": ranking["Release Year"],
                "加权得分": ranking["Weighted Score"].map(lambda v: f"{v:.3f}"),
            }
        )
        render_dark_table(
            display_ranking,
            table_class="ranking-table",
            card_class="ranking-table-card",
            html_columns={"排名", "平均评分"},
        )

    with catalog_tabs[4]:
        st.markdown('<div class="section-title">结果可视化</div>', unsafe_allow_html=True)
        chart_cols = st.columns(3)
        with chart_cols[0]:
            st.caption("评分分布")
            st.bar_chart(ratings["rating"].value_counts().sort_index())
        with chart_cols[1]:
            st.caption("用户评分行为")
            user_history = ratings.loc[ratings["user_id"] == user_id].copy()
            user_history["date"] = pd.to_datetime(user_history["timestamp"], unit="s").dt.date
            st.line_chart(user_history.groupby("date")["rating"].count())
        with chart_cols[2]:
            st.caption("推荐结果类型分布")
            chart_recs, _ = safe_personalized_recommendations(
                ratings,
                movies,
                avg_ratings,
                rating_counts,
                user_id,
                "Item-based Collaborative Filtering",
                10,
                20,
                "cosine",
                50,
            )
            st.bar_chart(recommendation_genre_counts(chart_recs))

    with catalog_tabs[5]:
        item_cf = load_item_cf_configured(ratings)
        title_opts = build_title_options(movies)
        selected_label = st.selectbox(
            "电影标题",
            options=title_opts["label"].tolist(),
            index=0,
            key="similar_movie",
        )
        selected_id = int(
            title_opts.loc[title_opts["label"] == selected_label, "movie_id"].iloc[0]
        )
        similar = item_cf.similar_items(selected_id, n=TOP_N)
        if similar.empty:
            st.warning(T["msg_not_enough_overlap"])
        else:
            recs = recommendations_table(similar, movies, avg_ratings)
            st.caption(f"当前电影ID：{selected_id}，显示最相似的 {TOP_N} 部电影。")
            render_recommendation_cards(recs, movies, ratings)


def render_favorites_page(account: dict, catalog: pd.DataFrame) -> None:
    """我的收藏：收藏海报墙 + 搜索添加。"""
    st.markdown('<div class="section-title">我的收藏</div>', unsafe_allow_html=True)
    account_id = int(account["account_id"])
    fav_ids = accounts.list_favorites(account_id)

    if fav_ids:
        fav_rows = catalog[catalog["Movie ID"].isin(fav_ids)].reset_index(drop=True)
        for start in range(0, len(fav_rows), 4):
            cols = st.columns(4)
            for col, (_, row) in zip(cols, fav_rows.iloc[start : start + 4].iterrows(), strict=False):
                movie_id = int(row["Movie ID"])
                with col:
                    render_html_block(movie_card_html(row))
                    if st.button("移除收藏", key=f"unfav_{movie_id}", use_container_width=True):
                        accounts.remove_favorite(account_id, movie_id)
                        st.rerun()
    else:
        st.info("暂无收藏，请在下方搜索并添加喜欢的电影。")

    st.markdown('<div class="section-title">添加收藏</div>', unsafe_allow_html=True)
    query = st.text_input("搜索电影标题或电影ID", key="fav_search")
    if query.strip():
        results = search_movies(catalog, query, T["label_all"]).head(8)
        if results.empty:
            st.info("未找到匹配电影。")
        for _, row in results.iterrows():
            movie_id = int(row["Movie ID"])
            title_zh = get_display_title(str(row["Title"]))
            row_cols = st.columns([0.75, 0.25])
            row_cols[0].write(title_zh)
            if movie_id in fav_ids:
                row_cols[1].caption("已收藏")
            elif row_cols[1].button("收藏", key=f"add_fav_{movie_id}"):
                accounts.add_favorite(account_id, movie_id)
                st.rerun()


def my_rating_card_html(
    movie_id: int,
    title_zh: str,
    genres_zh: str,
    year: object,
    rating: float,
    review: str,
    date_text: str,
) -> str:
    """Build a `.rating-grid-card` for one rated movie: poster, title, score, review, date."""
    rating_text = f"⭐ {int(rating)}/5"
    if review.strip():
        review_html = f'<div class="rating-grid-review">{html.escape(review)}</div>'
    else:
        review_html = '<div class="rating-grid-review empty">暂无评价内容</div>'
    return f"""
        <div class="rating-grid-card">
            {library_poster_html(movie_id, title_zh, title_zh, genres_zh, year, css_class="rating-grid-poster")}
            <div class="rating-grid-title">{html.escape(title_zh)}</div>
            <div class="rating-grid-meta">
                <span class="rating">{rating_text}</span>
                <span class="rating-grid-date">{html.escape(date_text)}</span>
            </div>
            {review_html}
        </div>
        """


def render_my_ratings_page(movies: pd.DataFrame, ml_user_id: int) -> None:
    """我的评分：展示已评分电影的海报、评分与评价内容，可通过“评价”弹窗修改。"""
    st.markdown('<div class="section-title">我的评分</div>', unsafe_allow_html=True)
    user_ratings = db.get_user_ratings(ml_user_id)
    if user_ratings.empty:
        st.info("暂无评分记录，去“电影库”页面为喜欢的电影打分吧。")
        return

    merged = user_ratings.merge(movies, on="movie_id", how="left").reset_index(drop=True)

    summary_cols = st.columns(3)
    summary_cols[0].metric("已评分电影数", f"{len(merged):,}")
    summary_cols[1].metric("平均评分", f"{merged['rating'].mean():.2f}")
    latest = pd.to_datetime(int(merged["timestamp"].max()), unit="s").strftime("%Y-%m-%d %H:%M")
    summary_cols[2].metric("最近评分时间", latest)

    for start in range(0, len(merged), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, merged.iloc[start : start + 4].iterrows(), strict=False):
            movie_id = int(row["movie_id"])
            title = str(row["title"])
            title_zh = get_display_title(title)
            genres_zh = translate_genres(movie_genre_text(row, movies))
            year = row.get("release_year")
            date_text = pd.to_datetime(int(row["timestamp"]), unit="s").strftime("%Y-%m-%d %H:%M")
            review = str(row["review"]) if not pd.isna(row["review"]) else ""
            with col:
                render_html_block(
                    my_rating_card_html(movie_id, title_zh, genres_zh, year, row["rating"], review, date_text)
                )
                if st.button("修改评价", key=f"edit_rating_{movie_id}", use_container_width=True):
                    show_rate_movie_dialog(movie_id, title_zh, ml_user_id)


def render_account_center_page(account: dict, ml_user_id: int) -> None:
    """个人中心：账号信息、统计数据、修改密码。"""
    st.markdown('<div class="section-title">个人中心</div>', unsafe_allow_html=True)
    n_ratings = int(len(db.get_user_ratings(ml_user_id)))
    n_favorites = len(accounts.list_favorites(int(account["account_id"])))

    render_html_block(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="value">{html.escape(account['username'])}</div><div class="label">用户名</div></div>
            <div class="stat-card"><div class="value">{html.escape(account['email'])}</div><div class="label">邮箱</div></div>
            <div class="stat-card"><div class="value">{account['created_at'][:10]}</div><div class="label">注册时间</div></div>
            <div class="stat-card"><div class="value">{n_ratings:,}</div><div class="label">评分总数</div></div>
        </div>
        <div class="stat-grid">
            <div class="stat-card"><div class="value">{n_favorites:,}</div><div class="label">收藏总数</div></div>
        </div>
        """
    )

    st.markdown('<div class="section-title">修改密码</div>', unsafe_allow_html=True)
    with st.form("change_password_form"):
        current = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm = st.text_input(T["label_confirm_password"], type="password")
        submitted = st.form_submit_button("更新密码")
    if submitted:
        if not accounts.verify_password(current, account["password_hash"]):
            st.error("当前密码不正确。")
        elif new_password != confirm:
            st.error(T["msg_password_mismatch"])
        elif not new_password:
            st.error("新密码不能为空。")
        else:
            accounts.update_password(int(account["account_id"]), new_password)
            st.success("密码已更新，下次登录请使用新密码。")


def render_profile_analytics_page(
    ratings: pd.DataFrame, movies: pd.DataFrame, catalog: pd.DataFrame, ml_user_id: int
) -> None:
    """我的画像：评分行为统计（类型偏好 / 年代偏好 / 平均评分）+ 注册偏好与推荐来源。"""
    st.markdown('<div class="section-title">我的画像</div>', unsafe_allow_html=True)

    account = st.session_state.get("account")
    preferred_genres_zh: list[str] = []
    if account is not None:
        prefs = accounts.get_preferences(account["account_id"])
        if prefs:
            preferred_genres_zh = prefs.get("genres", [])

    user_ratings = ratings.loc[ratings["user_id"] == ml_user_id]
    n_ratings = len(user_ratings)

    if n_ratings >= 5:
        source_label = T["profile_source_cf"]
    elif n_ratings > 0:
        source_label = T["profile_source_history"]
    elif preferred_genres_zh:
        source_label = T["profile_source_registration"]
    else:
        source_label = "暂无"

    preferred_genre_movie_count = 0
    if preferred_genres_zh:
        english_genres = [GENRE_EN_BY_ZH.get(g, g) for g in preferred_genres_zh if g]
        if english_genres:
            pattern = "|".join(re.escape(g) for g in english_genres)
            preferred_genre_movie_count = int(
                catalog["Genres"].str.contains(pattern, case=False, na=False, regex=True).sum()
            )

    render_html_block(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="value">{'、'.join(preferred_genres_zh) or '暂无'}</div><div class="label">{T['profile_preferred_genres']}</div></div>
            <div class="stat-card"><div class="value">{preferred_genre_movie_count:,}</div><div class="label">{T['profile_preferred_genre_movie_count']}</div></div>
            <div class="stat-card"><div class="value">{source_label}</div><div class="label">{T['profile_recommendation_source']}</div></div>
        </div>
        """
    )

    if preferred_genres_zh:
        pref_counts: dict[str, int] = {}
        for genre_zh in preferred_genres_zh:
            english_genre = GENRE_EN_BY_ZH.get(genre_zh, genre_zh)
            pref_counts[genre_zh] = int(
                catalog["Genres"].str.contains(re.escape(english_genre), case=False, na=False, regex=True).sum()
            )
        try:
            import plotly.express as px

            pref_df = pd.DataFrame(sorted(pref_counts.items(), key=lambda kv: kv[1]), columns=["类型", "电影数量"])
            fig = px.bar(pref_df, x="电影数量", y="类型", orientation="h", title="用户偏好类型分布")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#99aabb",
                height=max(240, 28 * len(pref_df)),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(pd.Series(pref_counts).sort_values())

    if user_ratings.empty:
        st.info("暂无评分数据，评分相关画像（类型偏好/偏好年代/平均评分）将在你评分后生成。")
        return

    merged = user_ratings.merge(movies, on="movie_id", how="left")
    genre_cols = display_genre_columns(movies)
    genre_counts = {g: int(merged[g].sum()) for g in genre_cols if g in merged.columns}
    genre_counts = {translate_genres(g): c for g, c in genre_counts.items() if c > 0}
    top_genres = sorted(genre_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    years = pd.to_numeric(merged.get("release_year"), errors="coerce").dropna()
    year_range = f"{int(years.min())}-{int(years.max())}" if not years.empty else "未知"
    avg_given = float(user_ratings["rating"].mean())

    render_html_block(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="value">{', '.join(g for g, _ in top_genres) or '暂无'}</div><div class="label">喜欢类型</div></div>
            <div class="stat-card"><div class="value">{year_range}</div><div class="label">偏好年代</div></div>
            <div class="stat-card"><div class="value">{avg_given:.2f} / 5</div><div class="label">平均评分</div></div>
            <div class="stat-card"><div class="value">{len(user_ratings):,}</div><div class="label">评分总数</div></div>
        </div>
        """
    )

    if genre_counts:
        try:
            import plotly.express as px

            genre_df = pd.DataFrame(sorted(genre_counts.items(), key=lambda kv: kv[1]), columns=["类型", "评分次数"])
            fig = px.bar(genre_df, x="评分次数", y="类型", orientation="h", title="类型偏好分布")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#99aabb",
                height=max(320, 28 * len(genre_df)),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(pd.Series(genre_counts).sort_values())

    accounts.save_profile(
        account_id=st.session_state["account"]["account_id"],
        top_genres=[g for g, _ in top_genres],
        year_range=year_range,
        preferred_min_rating=4.0 if avg_given >= 3.5 else 3.0,
    )


def admin_movies_state(movies: pd.DataFrame) -> pd.DataFrame:
    db.init_db(movies_df=movies)
    return db.get_movies_df()


def render_admin_module(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    catalog: pd.DataFrame,
) -> None:
    st.sidebar.markdown(f"### {T['nav_admin']}登录")
    username = st.sidebar.text_input(T["label_username"], value="admin", key="admin_login_username_input")
    password = st.sidebar.text_input(T["label_password"], type="password", key="admin_login_password_input")
    if not verify_admin(username, password):
        st.info(
            "请输入管理员账号密码以打开管理后台。"
            "除非在 .env 文件中通过 ADMIN_USERNAME / ADMIN_PASSWORD 覆盖，"
            "默认演示账号为 admin / admin123。"
        )
        return

    st.sidebar.success(T["msg_admin_verified"])
    admin_tabs = st.tabs(
        [
            T["nav_home"],
            T["nav_catalog"],
            T["nav_user_management"],
            T["nav_rating_records"],
            T["nav_algorithm_config"],
            T["nav_model_evaluation"],
            T["nav_system_stats"],
        ]
    )

    with admin_tabs[0]:
        st.markdown('<div class="section-title">管理后台首页</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="admin-subtitle">'
            "欢迎回来，以下是当前平台上最受欢迎的电影。完整的数据统计与图表请前往"
            f"“{T['nav_system_stats']}”标签页查看。"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">热门电影 Top 10</div>', unsafe_allow_html=True)
        render_hot_movies_top5(catalog, top_n=10)

    with admin_tabs[1]:
        st.markdown('<div class="section-title">电影数据管理</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-subtitle">浏览、搜索并维护电影库中的元数据记录。</div>', unsafe_allow_html=True)
        editable_movies = admin_movies_state(movies)
        admin_catalog = movie_catalog(editable_movies, avg_ratings, rating_counts)

        movie_dashboard_cols = st.columns(4)
        movie_dashboard_cols[0].metric("电影总数", f"{len(admin_catalog):,}")
        movie_dashboard_cols[1].metric("有评分电影数", f"{int((admin_catalog['Ratings'] > 0).sum()):,}")
        rated_avg = admin_catalog.loc[admin_catalog["Ratings"] > 0, "Average Rating"]
        movie_dashboard_cols[2].metric("平均评分", f"{rated_avg.mean():.2f}" if not rated_avg.empty else "暂无")
        movie_dashboard_cols[3].metric("类型数量", f"{len(display_genre_columns(editable_movies))}")

        filter_cols = st.columns(3)
        query = filter_cols[0].text_input(
            f"{T['btn_search']}电影（按ID、标题或类型）", key="admin_movie_search_input"
        )
        genre = filter_cols[1].selectbox(
            T["label_genre"], [T["label_all"], *genre_options_zh(editable_movies)], key="admin_movie_genre_select"
        )
        years = pd.to_numeric(admin_catalog["Release Year"], errors="coerce").dropna().astype(int)
        year_options = [T["label_all"], *sorted(years.unique().tolist(), reverse=True)]
        year_filter = filter_cols[2].selectbox("上映年份", year_options, key="admin_movie_year_select")

        filtered = search_movies(admin_catalog, query, genre)
        if year_filter != T["label_all"]:
            filtered = filtered[pd.to_numeric(filtered["Release Year"], errors="coerce") == year_filter]

        display_table = pd.DataFrame(
            {
                "电影ID": filtered["Movie ID"],
                "中文片名": filtered["Title"].map(get_display_title),
                "原片名": filtered["Title"],
                "类型": filtered["Genres"].map(translate_genres),
                "平均评分": filtered["Average Rating"].map(admin_rating_badge_html),
                "评分人数": filtered["Ratings"],
                "上映年份": filtered["Release Year"],
            }
        )
        render_admin_table(display_table, html_columns={"平均评分"})

        with st.expander("添加 / 编辑 / 删除电影记录"):
            action = st.radio(
                "操作",
                [T["btn_add_movie"], T["btn_edit_movie"], T["btn_delete_movie"]],
                horizontal=True,
                key="admin_movie_action_radio",
            )
            movie_id = st.number_input(T["label_movie_id"], min_value=1, value=1, step=1, key="admin_movie_id_input")
            title = st.text_input(T["label_title"], key="admin_movie_title_input")
            if st.button(T["btn_apply_change"], key="admin_movie_apply_button"):
                existing_ids = set(editable_movies["movie_id"].astype(int))
                if action == T["btn_add_movie"] and title:
                    if int(movie_id) in existing_ids:
                        st.error(f"电影ID {int(movie_id)} 已存在。")
                    else:
                        new_row = {col: 0 for col in editable_movies.columns}
                        new_row.update(
                            {
                                "movie_id": int(movie_id),
                                "title": title,
                                "release_date": "",
                                "video_release_date": "",
                                "imdb_url": "",
                                "release_year": "",
                            }
                        )
                        db.add_movie(new_row, administrator=username)
                        st.success(f"电影 #{int(movie_id)} 已添加并保存到数据库。")
                        st.rerun()
                elif action == T["btn_edit_movie"] and title:
                    if db.update_movie(int(movie_id), {"title": title}, administrator=username):
                        st.success(f"电影 #{int(movie_id)} 标题已更新并保存到数据库。")
                        st.rerun()
                    else:
                        st.error(f"未找到电影ID {int(movie_id)}。")
                elif action == T["btn_delete_movie"]:
                    if db.delete_movie(int(movie_id), administrator=username):
                        st.success(f"电影 #{int(movie_id)} 已从数据库删除。")
                        st.rerun()
                    else:
                        st.error(f"未找到电影ID {int(movie_id)}。")

        st.markdown("#### 管理员操作日志")
        st.dataframe(translate_columns(db.get_audit_log(50)), use_container_width=True, hide_index=True)

    with admin_tabs[2]:
        st.markdown('<div class="section-title">用户数据管理</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-subtitle">查看用户活跃度、评分行为与账号状态。</div>', unsafe_allow_html=True)
        user_stats = ratings.groupby("user_id").agg(
            Ratings=("rating", "count"),
            Average_Rating=("rating", "mean"),
            Last_Timestamp=("timestamp", "max"),
        )
        user_stats["Last Activity"] = pd.to_datetime(
            user_stats["Last_Timestamp"], unit="s"
        ).dt.strftime("%Y-%m-%d")
        user_stats = user_stats.drop(columns=["Last_Timestamp"]).reset_index()

        accounts_df = accounts.list_accounts()
        merged_users = user_stats.merge(
            accounts_df, left_on="user_id", right_on="movielens_user_id", how="left"
        ) if not accounts_df.empty else user_stats.assign(
            username=None, email=None, is_active=None, n_favorites=0
        )

        top_user_row = user_stats.loc[user_stats["Ratings"].idxmax()]
        user_summary_cols = st.columns(5)
        user_summary_cols[0].metric("用户总数", f"{len(merged_users):,}")
        user_summary_cols[1].metric("活跃用户数（评分数 ≥ 100）", f"{int((user_stats['Ratings'] >= 100).sum()):,}")
        user_summary_cols[2].metric("人均评分数", f"{user_stats['Ratings'].mean():.1f}")
        user_summary_cols[3].metric("新注册用户数", f"{len(accounts_df):,}")
        user_summary_cols[4].metric(
            "最活跃用户", f"用户#{int(top_user_row['user_id'])}", f"{int(top_user_row['Ratings']):,} 条评分"
        )

        search_cols = st.columns(2)
        user_id_query = search_cols[0].text_input(f"{T['btn_search']}用户ID", key="admin_user_search_input")
        username_query = search_cols[1].text_input(f"{T['btn_search']}用户名", key="admin_user_username_search_input")

        if user_id_query.strip():
            merged_users = merged_users[merged_users["user_id"].astype(str).str.contains(user_id_query.strip())]
        if username_query.strip():
            display_names = merged_users["username"].fillna(
                "MovieLens用户-" + merged_users["user_id"].astype(str)
            )
            merged_users = merged_users[display_names.str.contains(username_query.strip(), case=False, na=False)]

        display_users = pd.DataFrame(
            {
                "用户ID": merged_users["user_id"],
                "用户名": merged_users["username"].fillna("MovieLens用户-" + merged_users["user_id"].astype(str)),
                "邮箱": merged_users["email"].fillna("无"),
                "评分数量": merged_users["Ratings"],
                "收藏数": merged_users["n_favorites"].fillna(0).astype(int),
                "平均评分": merged_users["Average_Rating"].round(2),
                "最近活动日期": merged_users["Last Activity"],
                "状态": np.where(
                    merged_users["is_active"].isna(),
                    "历史数据用户",
                    np.where(merged_users["is_active"] == 1, "活跃账号", "已禁用"),
                ),
                "活跃等级": merged_users["Ratings"].map(activity_tier_badge_html),
            }
        )
        render_admin_table(
            display_users.sort_values("评分数量", ascending=False),
            html_columns={"活跃等级"},
        )

    with admin_tabs[3]:
        st.markdown('<div class="section-title">评分记录管理</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-subtitle">查询、筛选并审查全站评分记录。</div>', unsafe_allow_html=True)

        latest_ts = pd.to_datetime(ratings["timestamp"], unit="s").max()
        rating_summary_cols = st.columns(4)
        rating_summary_cols[0].metric("评分记录总数", f"{len(ratings):,}")
        rating_summary_cols[1].metric("平均评分", f"{ratings['rating'].mean():.2f}")
        rating_summary_cols[2].metric("最高评分人次", f"{int(ratings['rating'].value_counts().max()):,}")
        rating_summary_cols[3].metric("最新评分日期", latest_ts.strftime("%Y-%m-%d"))

        filters = st.columns(3)
        filter_user = filters[0].text_input(f"{T['btn_search']}用户ID", key="admin_rating_user_input")
        filter_movie = filters[1].text_input(f"{T['btn_search']}电影ID或标题", key="admin_rating_movie_input")
        filter_score = filters[2].selectbox(
            "筛选评分", [T["label_all"], 1, 2, 3, 4, 5], key="admin_rating_score_select"
        )
        rating_view = ratings.merge(movies[["movie_id", "title"]], on="movie_id", how="left")
        if filter_user.strip():
            rating_view = rating_view[rating_view["user_id"].astype(str).str.contains(filter_user.strip())]
        if filter_movie.strip():
            needle = filter_movie.strip()
            rating_view = rating_view[
                rating_view["movie_id"].astype(str).str.contains(needle)
                | rating_view["title"].str.contains(needle, case=False, na=False)
            ]
        if filter_score != T["label_all"]:
            rating_view = rating_view[rating_view["rating"] == int(filter_score)]

        st.metric("缺失值数量", int(ratings.isna().sum().sum()))
        st.metric("异常评分数量", int((~ratings["rating"].between(1, 5)).sum()))

        display_ratings = pd.DataFrame(
            {
                "用户ID": rating_view["user_id"],
                "电影ID": rating_view["movie_id"],
                "电影标题": rating_view["title"].map(get_display_title),
                "评分": rating_view["rating"].map(admin_rating_badge_html),
                "评分日期": pd.to_datetime(rating_view["timestamp"], unit="s").dt.strftime("%Y-%m-%d"),
            }
        )
        st.caption(f"共 {len(display_ratings):,} 条记录，最多显示前 500 条。")
        render_admin_table(display_ratings.head(500), html_columns={"评分"})

    with admin_tabs[4]:
        st.markdown('<div class="section-title">算法配置</div>', unsafe_allow_html=True)

        algorithm_options = [
            "Item-based Collaborative Filtering",
            "User-based Collaborative Filtering",
            "SVD Matrix Factorization",
            "Hybrid",
        ]
        current_algorithm = st.session_state.get("admin_algorithm", algorithm_options[0])
        st.info(f"当前默认算法：**{current_algorithm}**")

        st.session_state["admin_algorithm"] = st.selectbox(
            "默认算法",
            algorithm_options,
            index=algorithm_options.index(current_algorithm) if current_algorithm in algorithm_options else 0,
            key="admin_model_algorithm_select",
        )
        st.session_state["admin_k"] = st.slider(
            T["label_neighbors_k"], 5, 50, int(st.session_state.get("admin_k", 20)), 5, key="admin_neighbors_k_slider"
        )
        st.session_state["admin_metric"] = st.selectbox(
            T["label_similarity_method"], ["cosine", "pearson"], key="admin_similarity_method_select"
        )
        st.session_state["admin_factors"] = st.slider(
            T["label_svd_factors"], 10, 80, int(st.session_state.get("admin_factors", 50)), 10, key="admin_svd_factors_slider"
        )
        st.session_state["admin_rec_count"] = st.slider(
            T["label_recommendation_count"], 5, 30, int(st.session_state.get("admin_rec_count", 10)), 5, key="admin_rec_count_slider"
        )
        if st.button(T["btn_save_config"], key="admin_save_config_button"):
            st.success("当前算法配置已保存到会话状态。")

        st.markdown("#### 算法说明")
        explain_cols = st.columns(4)
        explanations = [
            ("UserCF", "基于相似用户", "找到与目标用户兴趣相似的其他用户，根据这些用户的评分加权预测目标用户对电影的偏好。"),
            ("ItemCF", "基于相似电影", "根据电影之间的评分相似度，为用户推荐与其历史喜欢的电影相似的其他电影。"),
            ("SVD", "矩阵分解", "将用户-电影评分矩阵分解为隐因子矩阵，通过隐因子向量的内积预测缺失评分。"),
            ("Hybrid", "混合推荐", "结合协同过滤与电影元数据特征（类型、用户/电影统计特征），用回归模型综合预测评分。"),
        ]
        for col, (name, subtitle, desc) in zip(explain_cols, explanations):
            with col:
                render_html_block(
                    f"""
                    <div class="capability-card">
                        <h4>{name}</h4>
                        <p><strong>{subtitle}</strong></p>
                        <p>{desc}</p>
                    </div>
                    """
                )

    with admin_tabs[5]:
        st.markdown('<div class="section-title">模型评估</div>', unsafe_allow_html=True)
        st.caption("在按用户时间序列划分的留出测试集上评估各算法的评分预测误差与 Top-10 排序质量（结果已缓存，刷新页面不会重新训练）。")
        if st.button(T["btn_run_evaluation"], key="admin_run_evaluation_button"):
            ranking_evaluation_summary.clear()
        eval_table = ranking_evaluation_summary(ratings, movies)
        st.session_state["eval_table"] = eval_table

        best = eval_table.loc[eval_table["RMSE"].idxmin()]
        metric_cols = st.columns(6)
        metric_cols[0].metric("最优 RMSE", f"{best['RMSE']:.3f}", best["Algorithm"])
        metric_cols[1].metric("最优 MAE", f"{best['MAE']:.3f}", best["Algorithm"])
        best_p = eval_table.loc[eval_table["Precision@10"].idxmax()]
        metric_cols[2].metric("最优 Precision@10", f"{best_p['Precision@10']:.3f}", best_p["Algorithm"])
        best_r = eval_table.loc[eval_table["Recall@10"].idxmax()]
        metric_cols[3].metric("最优 Recall@10", f"{best_r['Recall@10']:.3f}", best_r["Algorithm"])
        best_h = eval_table.loc[eval_table["HitRate@10"].idxmax()]
        metric_cols[4].metric("最优 HitRate@10", f"{best_h['HitRate@10']:.3f}", best_h["Algorithm"])
        best_n = eval_table.loc[eval_table["NDCG@10"].idxmax()]
        metric_cols[5].metric("最优 NDCG@10", f"{best_n['NDCG@10']:.3f}", best_n["Algorithm"])

        st.dataframe(translate_columns(eval_table), use_container_width=True, hide_index=True)

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**预测误差对比（越低越好）**")
            st.bar_chart(eval_table.set_index("Algorithm")[["MAE", "RMSE"]])
        with chart_cols[1]:
            st.markdown("**Top-10 排序质量对比（越高越好）**")
            st.bar_chart(eval_table.set_index("Algorithm")[["Precision@10", "Recall@10", "HitRate@10", "NDCG@10"]])

    with admin_tabs[6]:
        st.markdown('<div class="section-title">系统统计仪表盘</div>', unsafe_allow_html=True)
        if st.button(T["btn_refresh_data"], key="admin_stats_refresh_button"):
            st.rerun()

        try:
            conn = db.get_connection()
            conn.execute("SELECT 1")
            conn.close()
            db_status = "正常"
        except Exception:
            db_status = "异常"

        sparsity = 1 - len(ratings) / (ratings["user_id"].nunique() * movies["movie_id"].nunique())
        local_poster_count = len(build_local_poster_index()["exact"])

        health_cols = st.columns(6)
        health_cols[0].metric("MovieLens用户数", f"{ratings['user_id'].nunique():,}")
        health_cols[1].metric("电影总数", f"{movies['movie_id'].nunique():,}")
        health_cols[2].metric("评分总数", f"{len(ratings):,}")
        health_cols[3].metric("数据稀疏度", f"{sparsity * 100:.2f}%")
        health_cols[4].metric("本地海报数量", f"{local_poster_count:,}")
        health_cols[5].metric("数据库状态", db_status)

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**评分分布**")
            st.bar_chart(ratings["rating"].value_counts().sort_index())
        with chart_cols[1]:
            st.markdown("**电影类型分布**")
            genre_totals = {
                genre: int(movies[genre].sum())
                for genre in display_genre_columns(movies)
                if genre in movies
            }
            st.bar_chart(pd.Series(genre_totals).sort_values(ascending=False))

        st.markdown("**热门电影 Top10**")
        st.dataframe(
            translate_columns(popular_movies(catalog, "Weighted Score", 10)),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**用户活跃度分布**")
        rating_counts_per_user = ratings.groupby("user_id")["rating"].count()
        activity_bins = pd.cut(
            rating_counts_per_user,
            bins=[0, 20, 50, 100, 200, 500, float("inf")],
            labels=["1-20", "21-50", "51-100", "101-200", "201-500", "500+"],
        )
        st.bar_chart(activity_bins.value_counts().sort_index())


def main() -> None:
    st.set_page_config(
        page_title="FilmTrace | 电影推荐系统",
        page_icon=":movie_camera:",
        layout="wide",
    )
    inject_theme()

    try:
        ratings, movies = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("请解压 MovieLens 100K 数据集，确保 `data/raw/ml-100k/u.data` 存在，然后重启应用。")
        return

    avg_ratings = movie_avg_ratings(ratings)
    rating_counts = movie_rating_counts(ratings)
    catalog = movie_catalog(movies, avg_ratings, rating_counts)
    accounts.ensure_schema()

    entry = st.session_state.get("entry", "landing")

    if entry == "landing":
        render_header()
        render_landing_page(ratings, movies)
        return

    render_header()
    st.sidebar.markdown(
        '<div class="sidebar-brand">'
        '<span class="sidebar-brand-icon">🎬</span>'
        '<span>Film<span class="sidebar-brand-trace">Trace</span></span>'
        "</div>"
        '<div class="sidebar-subtitle">智能电影推荐平台</div>',
        unsafe_allow_html=True,
    )

    if entry == "admin":
        if st.sidebar.button(T["btn_back_to_landing"], key="sidebar_back_to_landing_button"):
            st.session_state["entry"] = "landing"
            st.session_state.pop("account", None)
            st.rerun()
        render_admin_module(ratings, movies, avg_ratings, rating_counts, catalog)
        return

    # entry == "user"
    account = st.session_state.get("account")
    if account is None:
        if st.sidebar.button(T["btn_back_to_landing"], key="sidebar_back_to_landing_button"):
            st.session_state["entry"] = "landing"
            st.session_state.pop("account", None)
            st.rerun()
        render_auth_page(ratings)
        return

    ml_user_id = int(account["movielens_user_id"] or assign_movielens_user_id(ratings, account["account_id"]))

    st.sidebar.markdown(
        '<div class="sidebar-user-card">'
        f'<div class="sidebar-user-name">{html.escape(account["username"])}</div>'
        f'<div class="sidebar-user-status">已登录</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    back_col, logout_col = st.sidebar.columns(2)
    with back_col:
        if st.button(T["btn_back_to_landing"], key="sidebar_back_to_landing_button"):
            st.session_state["entry"] = "landing"
            st.session_state.pop("account", None)
            st.rerun()
    with logout_col:
        if st.button(T["btn_logout"]):
            st.session_state.pop("account", None)
            st.rerun()

    nav_sections = [
        (
            T["sidebar_section_discover"],
            [T["nav_user_home"], T["nav_user_for_you"], T["nav_user_catalog"]],
        ),
        (
            T["sidebar_section_personal"],
            [
                T["nav_user_favorites"],
                T["nav_user_ratings"],
                T["nav_user_profile_analytics"],
                T["nav_user_account"],
            ],
        ),
    ]

    if "user_nav_page" not in st.session_state:
        st.session_state["user_nav_page"] = T["nav_user_home"]
    current_page = st.session_state["user_nav_page"]

    for section_label, items in nav_sections:
        st.sidebar.markdown(f'<div class="sidebar-section-label">{section_label}</div>', unsafe_allow_html=True)
        active_index = items.index(current_page) if current_page in items else None
        selection = st.sidebar.radio(
            section_label,
            items,
            index=active_index,
            label_visibility="collapsed",
            key=f"nav_radio_{section_label}_{active_index}",
        )
        if selection is not None and selection != current_page:
            st.session_state["user_nav_page"] = selection
            st.rerun()

    page = st.session_state["user_nav_page"]
    if page == T["nav_user_home"]:
        render_home_page(ratings, movies, avg_ratings, rating_counts, catalog, ml_user_id)
    elif page == T["nav_user_for_you"]:
        render_for_you_page(ratings, movies, avg_ratings, rating_counts, catalog, ml_user_id)
    elif page == T["nav_user_catalog"]:
        render_catalog_page(ratings, movies, avg_ratings, rating_counts, catalog, ml_user_id)
    elif page == T["nav_user_favorites"]:
        render_favorites_page(account, catalog)
    elif page == T["nav_user_ratings"]:
        render_my_ratings_page(movies, ml_user_id)
    elif page == T["nav_user_profile_analytics"]:
        render_profile_analytics_page(ratings, movies, catalog, ml_user_id)
    else:
        render_account_center_page(account, ml_user_id)


if __name__ == "__main__":
    main()
