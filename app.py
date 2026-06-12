"""Streamlit demo for rating prediction and similar-movie recommendation."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db
from src import recsys_service as recsys
from src.auth import verify_admin
from src.baselines import BiasBaseline
from src.data_cleaning import align_ratings_with_movies, clean_movies, clean_ratings
from src.data_loader import load_movies, load_ratings
from src.i18n import GENRE_EN_BY_ZH, GENRE_ZH, T, get_display_title, translate_columns, translate_genres
from src.item_based_cf import ItemBasedCF
from src.metrics import mae, rmse
from src.posters import fetch_poster_url, list_local_posters, local_poster_path
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
def load_item_cf(ratings: pd.DataFrame) -> ItemBasedCF:
    model = ItemBasedCF()
    model.fit(ratings)
    return model


@st.cache_resource(show_spinner="正在训练基于用户的协同过滤模型...")
def load_user_cf(ratings: pd.DataFrame, k: int = 20, metric: str = "cosine") -> UserBasedCF:
    model = UserBasedCF(k=k, metric=metric)
    model.fit(ratings)
    return model


@st.cache_resource(show_spinner="正在训练基于物品的协同过滤模型...")
def load_item_cf_configured(
    ratings: pd.DataFrame,
    k: int = 20,
    metric: str = "cosine",
) -> ItemBasedCF:
    model = ItemBasedCF(k=k, metric=metric)
    model.fit(ratings)
    return model


@st.cache_resource(show_spinner="正在训练 SVD 矩阵分解模型...")
def load_svd_model(ratings: pd.DataFrame, factors: int = 50) -> SVDRecommender:
    model = SVDRecommender(n_components=factors)
    model.fit(ratings)
    return model


@st.cache_resource(show_spinner="正在训练评分预测模型...")
def load_bias_model(ratings: pd.DataFrame) -> BiasBaseline:
    model = BiasBaseline(n_epochs=20, reg=1.0)
    model.fit(ratings)
    return model


@st.cache_data
def movie_avg_ratings(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("movie_id")["rating"].mean()


@st.cache_data
def movie_rating_counts(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("movie_id")["rating"].count()


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


def catalog_display_table(results: pd.DataFrame) -> pd.DataFrame:
    """Build a Chinese-column display table from an English-column catalog slice."""
    return pd.DataFrame(
        {
            "电影ID": results["Movie ID"],
            "中文片名": results["Title"].map(get_display_title),
            "原片名": results["Title"],
            "类型": results["Genres"].map(translate_genres),
            "平均评分": results["Average Rating"],
            "评分人数": results["Ratings"],
            "上映年份": results["Release Year"],
        }
    )


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


def calibrated_recommendation_score(
    raw_pred: float,
    avg_rating: float,
    user_mean: float,
    confidence: float,
    rating_count: int,
    global_mean: float,
) -> float:
    return recsys.calibrated_score(raw_pred, avg_rating, user_mean, confidence, rating_count, global_mean)


@st.cache_data(show_spinner=False)
def personalized_recommendations(
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
) -> pd.DataFrame:
    model = algorithm_model(ratings, algorithm, k, metric, factors)
    rated = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])
    candidates = [int(mid) for mid in movies["movie_id"] if int(mid) not in rated]
    user_mean = float(ratings.loc[ratings["user_id"] == user_id, "rating"].mean())
    global_mean = float(ratings["rating"].mean())
    max_count = max(1, int(rating_counts.max()))
    rows = []
    for movie_id in candidates:
        raw_pred = float(model.predict(user_id, movie_id))
        movie_row = movies.loc[movies["movie_id"] == movie_id].iloc[0]
        genres = movie_genre_text(movie_row, movies)
        avg_rating = float(avg_ratings.get(movie_id, global_mean))
        rating_count = int(rating_counts.get(movie_id, 0))
        confidence = min(1.0, np.log1p(rating_count) / np.log1p(max_count))
        score = calibrated_recommendation_score(
            raw_pred,
            avg_rating,
            user_mean,
            confidence,
            rating_count,
            global_mean,
        )
        rows.append(
            {
                "Movie ID": movie_id,
                "Title": str(movie_row["title"]),
                "Recommendation Score": score,
                "Raw Model Score": raw_pred,
                "Average Rating": avg_rating,
                "Ratings": rating_count,
                "Confidence": confidence,
                "Genres": genres,
                "Release Year": movie_row.get("release_year", ""),
                "Reason": recommendation_reason(algorithm, genres),
            }
        )
    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs
    recs = recs.sort_values(
        ["Recommendation Score", "Confidence", "Average Rating", "Ratings"],
        ascending=False,
    ).head(top_n)
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
) -> pd.DataFrame:
    """Compute personalized recommendations with a spinner and a Chinese error message."""
    try:
        with st.spinner("正在生成个性化推荐..."):
            return personalized_recommendations(
                ratings, movies, avg_ratings, rating_counts, user_id, algorithm, top_n, k, metric, factors,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"生成推荐时发生错误，请尝试调整参数后重试。错误信息：{exc}")
        return pd.DataFrame()


def recommendation_genre_counts(recs: pd.DataFrame) -> pd.Series:
    counts: dict[str, int] = {}
    for genres in recs.get("Genres", []):
        for genre in str(genres).split(","):
            genre = genre.strip()
            if genre and genre != "未知类型":
                genre_zh = GENRE_ZH.get(genre, genre)
                counts[genre_zh] = counts.get(genre_zh, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


@st.cache_data(show_spinner="正在采样留出集上评估模型...")
def evaluation_summary(ratings: pd.DataFrame) -> pd.DataFrame:
    train, test = user_temporal_split(ratings)
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
    genres = recsys.genre_text(movie_row, genre_columns(movies))
    return genres if genres != "Unknown" else "未知类型"


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


def genre_columns(movies: pd.DataFrame) -> list[str]:
    return recsys.genre_columns(movies)


def genre_options_zh(movies: pd.DataFrame) -> list[str]:
    """Chinese genre names for filter selectboxes, derived from genre columns."""
    return [GENRE_ZH.get(genre, genre) for genre in genre_columns(movies)]


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
    genres = genre_columns(movies)
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
            max-width: 1160px;
            padding-top: 2.2rem;
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
                linear-gradient(90deg, rgba(20, 24, 28, 0.98) 0%, rgba(20, 24, 28, 0.72) 46%, rgba(20, 24, 28, 0.96) 100%),
                radial-gradient(circle at 74% 28%, rgba(255, 153, 51, 0.22), transparent 13rem),
                linear-gradient(135deg, #202830 0%, #14181c 58%, #0f1216 100%);
            box-shadow: rgba(0, 0, 0, 0.32) 0 18px 50px;
            border: 1px solid rgba(221, 238, 255, 0.06);
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: auto 2rem 0 2rem;
            height: 110px;
            background:
                linear-gradient(90deg, #2e1d40, #18324d, #511d1d, #17482e, #2d2d18, #304150);
            opacity: 0.78;
            clip-path: polygon(0 18%, 15% 0, 30% 22%, 45% 4%, 60% 24%, 75% 8%, 100% 20%, 100% 100%, 0 100%);
            filter: saturate(0.85);
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

        .stButton > button {
            background: var(--green);
            color: var(--white);
            border: 0;
            border-radius: 4px;
            font-weight: 800;
        }

        .stButton > button:hover {
            background: var(--vivid);
            color: var(--white);
            border: 0;
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

        @media (max-width: 760px) {
            .nav-links { display: none; }
            .hero { min-height: 430px; padding: 1.2rem; }
            .hero h1 { font-size: 1.9rem; }
            .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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


def render_hero(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    render_html_block(
        f"""
        <section class="hero">
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


def render_featured_movies(catalog: pd.DataFrame, max_items: int = 8) -> None:
    """Showcase movies straight from the local `电影照片/` poster gallery.

    For posters whose filename (Chinese title) matches a MovieLens movie via
    `get_display_title`, the real genre/release year from the catalog is
    shown; otherwise the poster is shown with its title only.
    """
    posters = list_local_posters()
    if not posters:
        st.info("未找到本地海报文件，请检查 `电影照片/` 目录。")
        return
    zh_titles = catalog["Title"].map(get_display_title)
    items = list(posters.items())[:max_items]
    for start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (title_zh, path) in zip(cols, items[start : start + 4], strict=False):
            match = catalog[zh_titles == title_zh]
            with col:
                with st.container(border=True):
                    st.image(str(path), use_container_width=True)
                    st.markdown(f"**{title_zh}**")
                    if not match.empty:
                        row = match.iloc[0]
                        genres_zh = translate_genres(str(row["Genres"]))
                        year = row.get("Release Year", "")
                        st.caption(f"{genres_zh}　{year}")
                    else:
                        st.caption("电影海报展示")


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
    if imdb_url and imdb_url != "nan":
        st.link_button(T["btn_open_imdb"], imdb_url)
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
        if imdb_url and imdb_url != "nan":
            st.link_button(T["btn_open_imdb"], imdb_url)


def render_user_summary(ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int) -> None:
    history_df = user_history_table(ratings, movies, user_id)
    genre_pref = user_genre_preferences(ratings, movies, user_id)
    top_genre = (
        translate_genres(str(genre_pref.iloc[0]["Genre"])) if not genre_pref.empty else "暂无数据"
    )
    liked_count = int((history_df["User Rating"] >= 4).sum())
    cols = st.columns(5)
    cols[0].metric("已评分电影数", f"{len(history_df):,}")
    cols[1].metric("平均评分", f"{history_df['User Rating'].mean():.2f}")
    cols[2].metric("偏好类型", top_genre)
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

    st.markdown('<div class="section-title">精选电影</div>', unsafe_allow_html=True)
    st.caption("来自本地海报库的精选电影展示。")
    render_featured_movies(catalog)

    st.markdown('<div class="section-title">热门电影</div>', unsafe_allow_html=True)
    st.caption("MovieLens 100K 中综合评分最高、最受欢迎的电影。")
    render_catalog_cards(popular_movies(catalog, "Weighted Score", 10))

    st.markdown('<div class="section-title">数据集概览</div>', unsafe_allow_html=True)
    render_dataset_stats(ratings, movies)

    st.markdown('<div class="section-title">系统能力</div>', unsafe_allow_html=True)
    render_capability_cards()


def render_for_you_page(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    avg_ratings: pd.Series,
    rating_counts: pd.Series,
    user_id: int,
) -> None:
    render_user_summary(ratings, movies, user_id)
    st.markdown('<div class="section-title">为你推荐</div>', unsafe_allow_html=True)
    st.caption(f"基于用户 #{user_id} 的历史评分，系统已自动生成以下个性化推荐结果。")

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

    recs = safe_personalized_recommendations(
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
    render_personalized_cards(recs)

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
        search_cols = st.columns([0.65, 0.35])
        query = search_cols[0].text_input(f"{T['btn_search']}（按标题或电影ID）")
        genre = search_cols[1].selectbox(T["label_genre"], [T["label_all"], *genre_options_zh(movies)])

        results = search_movies(catalog, query, genre)
        if results.empty:
            st.info("未找到匹配电影，请尝试其他关键词。")
        else:
            is_default_view = not query.strip() and genre == T["label_all"]
            display_results = results.head(20) if is_default_view else results.head(200)
            if is_default_view:
                st.caption("当前显示热门高分电影 Top 20，输入关键词或选择类型可进一步筛选。")
            st.dataframe(
                catalog_display_table(display_results),
                use_container_width=True,
                hide_index=True,
            )

    with catalog_tabs[1]:
        st.markdown('<div class="section-title">电影详情页</div>', unsafe_allow_html=True)
        render_movie_detail_page(catalog, movies)

    with catalog_tabs[2]:
        st.markdown('<div class="section-title">用户评分记录</div>', unsafe_allow_html=True)
        st.caption("MovieLens 100K 数据集记录了用户的显式评分及对应的 Unix 时间戳。")
        st.dataframe(
            translate_columns(user_history_table(ratings, movies, user_id)),
            use_container_width=True,
            hide_index=True,
        )
        genre_pref = user_genre_preferences(ratings, movies, user_id)
        st.markdown('<div class="section-title">偏好分析</div>', unsafe_allow_html=True)
        st.dataframe(translate_columns(genre_pref), use_container_width=True, hide_index=True)

    with catalog_tabs[3]:
        st.markdown('<div class="section-title">热门电影排行</div>', unsafe_allow_html=True)
        rank_cols = st.columns(2)
        rank_by = rank_cols[0].selectbox(
            T["label_rank_by"],
            ["Weighted Score", "Average Rating", "Number of Ratings"],
        )
        pop_n = rank_cols[1].selectbox(T["label_show_top"], [10, 20], index=0)
        st.dataframe(
            translate_columns(popular_movies(catalog, rank_by, int(pop_n))),
            use_container_width=True,
            hide_index=True,
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
            chart_recs = safe_personalized_recommendations(
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
    username = st.sidebar.text_input(T["label_username"], value="admin")
    password = st.sidebar.text_input(T["label_password"], type="password")
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
        st.markdown('<div class="section-title">数据集规模</div>', unsafe_allow_html=True)
        render_dataset_stats(ratings, movies)

        active_users = int((ratings.groupby("user_id")["rating"].count() >= 100).sum())
        overview_cols = st.columns(3)
        overview_cols[0].metric("评分总数", f"{len(ratings):,}")
        overview_cols[1].metric("活跃用户数（评分数 ≥ 100）", f"{active_users:,}")
        sparsity = 1 - len(ratings) / (
            ratings["user_id"].nunique() * movies["movie_id"].nunique()
        )
        overview_cols[2].metric("数据稀疏度", f"{sparsity * 100:.2f}%")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**评分分布**")
            st.bar_chart(ratings["rating"].value_counts().sort_index())
        with chart_cols[1]:
            st.markdown("**电影类型分布**")
            genre_totals = {
                genre: int(movies[genre].sum())
                for genre in genre_columns(movies)
                if genre in movies
            }
            st.bar_chart(pd.Series(genre_totals).sort_values(ascending=False))

        st.markdown("**热门电影 Top 5**")
        render_catalog_cards(popular_movies(catalog, "Weighted Score", 5))

        st.markdown("**模型对比（RMSE / MAE）**")
        st.caption("点击下方“模型评估”标签页中的“运行评估”按钮可刷新此对比结果。")
        admin_eval_table = st.session_state.get("eval_table")
        if admin_eval_table is None:
            st.info("尚未运行模型评估，请前往“模型评估”标签页运行评估。")
        else:
            st.dataframe(translate_columns(admin_eval_table), use_container_width=True, hide_index=True)
            st.bar_chart(admin_eval_table.set_index("Algorithm")[["MAE", "RMSE"]])

    with admin_tabs[1]:
        st.markdown('<div class="section-title">电影数据管理</div>', unsafe_allow_html=True)
        editable_movies = admin_movies_state(movies)
        admin_catalog = movie_catalog(editable_movies, avg_ratings, rating_counts)
        query = st.text_input(f"{T['btn_search']}电影（按ID、标题或类型）", key="admin_movie_search")
        genre = st.selectbox(T["label_genre"], [T["label_all"], *genre_options_zh(editable_movies)], key="admin_movie_genre")
        st.dataframe(
            translate_columns(search_movies(admin_catalog, query, genre)),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("添加 / 编辑 / 删除电影记录"):
            action = st.radio("操作", [T["btn_add_movie"], T["btn_edit_movie"], T["btn_delete_movie"]], horizontal=True)
            movie_id = st.number_input(T["label_movie_id"], min_value=1, value=1, step=1, key="admin_movie_id")
            title = st.text_input(T["label_title"], key="admin_movie_title")
            if st.button(T["btn_apply_change"]):
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
        user_stats = ratings.groupby("user_id").agg(
            Ratings=("rating", "count"),
            Average_Rating=("rating", "mean"),
            Last_Timestamp=("timestamp", "max"),
        )
        user_stats["Last Activity"] = pd.to_datetime(
            user_stats["Last_Timestamp"], unit="s"
        ).dt.strftime("%Y-%m-%d")
        user_stats = user_stats.drop(columns=["Last_Timestamp"]).reset_index()
        user_stats["Activity Tier"] = np.where(user_stats["Ratings"] >= 100, "活跃", "不活跃")
        user_query = st.text_input(f"{T['btn_search']}用户ID")
        if user_query.strip():
            user_stats = user_stats[user_stats["user_id"].astype(str).str.contains(user_query.strip())]
        st.dataframe(translate_columns(user_stats.sort_values("Ratings", ascending=False)), use_container_width=True, hide_index=True)

    with admin_tabs[3]:
        st.markdown('<div class="section-title">评分数据管理</div>', unsafe_allow_html=True)
        filters = st.columns(3)
        filter_user = filters[0].text_input(f"{T['btn_search']}用户ID")
        filter_movie = filters[1].text_input(f"{T['btn_search']}电影ID")
        filter_score = filters[2].selectbox("筛选评分", [T["label_all"], 1, 2, 3, 4, 5])
        rating_view = ratings.copy()
        if filter_user.strip():
            rating_view = rating_view[rating_view["user_id"].astype(str).str.contains(filter_user.strip())]
        if filter_movie.strip():
            rating_view = rating_view[rating_view["movie_id"].astype(str).str.contains(filter_movie.strip())]
        if filter_score != T["label_all"]:
            rating_view = rating_view[rating_view["rating"] == int(filter_score)]
        st.metric("缺失值数量", int(ratings.isna().sum().sum()))
        st.metric("异常评分数量", int((~ratings["rating"].between(1, 5)).sum()))
        st.dataframe(translate_columns(rating_view.head(500)), use_container_width=True, hide_index=True)

    with admin_tabs[4]:
        st.markdown('<div class="section-title">算法配置</div>', unsafe_allow_html=True)
        st.session_state["admin_algorithm"] = st.selectbox(
            "当前算法",
            [
                "Item-based Collaborative Filtering",
                "User-based Collaborative Filtering",
                "SVD Matrix Factorization",
            ],
            index=0,
        )
        st.session_state["admin_k"] = st.slider(T["label_neighbors_k"], 5, 50, int(st.session_state.get("admin_k", 20)), 5)
        st.session_state["admin_factors"] = st.slider(T["label_svd_factors"], 10, 80, int(st.session_state.get("admin_factors", 50)), 10)
        st.session_state["admin_metric"] = st.selectbox(T["label_similarity_method"], ["cosine", "pearson"])
        if st.button(T["btn_save_config"]):
            st.success("当前算法配置已保存到会话状态。")

    with admin_tabs[5]:
        st.markdown('<div class="section-title">模型评估</div>', unsafe_allow_html=True)
        if st.button(T["btn_run_evaluation"]):
            st.session_state["eval_table"] = evaluation_summary(ratings)
        eval_table = st.session_state.get("eval_table")
        if eval_table is None:
            st.info("点击按钮，在采样的时序留出测试集上评估 MAE 和 RMSE。")
        else:
            st.dataframe(translate_columns(eval_table), use_container_width=True, hide_index=True)
            st.bar_chart(eval_table.set_index("Algorithm")[["MAE", "RMSE"]])

    with admin_tabs[6]:
        st.markdown('<div class="section-title">系统统计仪表盘</div>', unsafe_allow_html=True)
        if st.button(T["btn_refresh_data"]):
            st.rerun()
        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.caption("评分数最多的电影")
            st.dataframe(translate_columns(popular_movies(catalog, "Number of Ratings", 20)), use_container_width=True, hide_index=True)
        with stat_cols[1]:
            st.caption("平均评分最高的电影")
            st.dataframe(translate_columns(popular_movies(catalog, "Average Rating", 20)), use_container_width=True, hide_index=True)
        genre_totals = {
            genre: int(movies[genre].sum())
            for genre in genre_columns(movies)
            if genre in movies
        }
        st.caption("电影类型分布")
        st.bar_chart(pd.Series(genre_totals).sort_values(ascending=False))


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

    render_header()

    st.sidebar.markdown("### 用户登录")
    user_id = st.sidebar.number_input(
        T["label_user_id"],
        min_value=int(ratings["user_id"].min()),
        max_value=int(ratings["user_id"].max()),
        value=int(st.session_state.get("active_user_id", ratings["user_id"].min())),
        step=1,
    )
    st.session_state["active_user_id"] = int(user_id)
    st.sidebar.success(f"已加载用户 {int(user_id)}")

    page = st.sidebar.radio(
        T["entrance_select"],
        [T["nav_home"], T["nav_for_you"], T["nav_catalog"], T["nav_admin"]],
    )
    if page == T["nav_home"]:
        render_home_page(ratings, movies, avg_ratings, rating_counts, catalog, int(user_id))
    elif page == T["nav_for_you"]:
        render_for_you_page(ratings, movies, avg_ratings, rating_counts, int(user_id))
    elif page == T["nav_catalog"]:
        render_catalog_page(ratings, movies, avg_ratings, rating_counts, catalog, int(user_id))
    else:
        render_admin_module(ratings, movies, avg_ratings, rating_counts, catalog)


if __name__ == "__main__":
    main()
