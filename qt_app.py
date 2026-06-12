"""PyQt5 desktop interface for the MovieLens recommendation system."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src import db
from src import recsys_service as recsys
from src.analysis import (
    paired_error_test,
    recommend_top_k_from_model,
    relevant_items_by_user,
    sampled_candidate_items_by_user,
)
from src.auth import verify_admin
from src.data_cleaning import align_ratings_with_movies, clean_movies, clean_ratings
from src.data_loader import load_movies, load_ratings
from src.i18n import T, translate_columns
from src.item_based_cf import ItemBasedCF
from src.metrics import evaluate_topk, mae, rmse
from src.preprocessing import user_temporal_split
from src.svd_model import SVDRecommender
from src.user_based_cf import UserBasedCF

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_POSTER_DIR = PROJECT_ROOT / "assets" / "posters"
LOCAL_COVER_IMAGES = sorted(LOCAL_POSTER_DIR.glob("*.jpg")) + sorted(
    LOCAL_POSTER_DIR.glob("*.png")
) + sorted(LOCAL_POSTER_DIR.glob("*.jpeg"))
NON_GENRE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
}


def set_table_data(table: QTableWidget, data: pd.DataFrame, max_rows: int = 300) -> None:
    """Fill a QTableWidget from a dataframe."""
    view = translate_columns(data.head(max_rows).copy())
    table.clear()
    table.setRowCount(len(view))
    table.setColumnCount(len(view.columns))
    table.setHorizontalHeaderLabels([str(col) for col in view.columns])
    for row_idx, row in enumerate(view.itertuples(index=False)):
        for col_idx, value in enumerate(row):
            item = QTableWidgetItem("" if pd.isna(value) else str(value))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            item.setToolTip("" if pd.isna(value) else str(value))
            table.setItem(row_idx, col_idx, item)
    table.setWordWrap(True)
    table.resizeColumnsToContents()
    compact_headers = {
        "电影ID": "电影ID",
        "平均评分": "评分",
        "Average_Rating": "评分",
        "上映年份": "年份",
        "评分数": "数量",
    }
    for col_idx, col_name in enumerate(view.columns):
        if col_name in compact_headers:
            table.horizontalHeaderItem(col_idx).setText(compact_headers[col_name])
    if "推荐理由" in view.columns:
        reason_col = list(view.columns).index("推荐理由")
        table.setColumnWidth(reason_col, 360)
        table.resizeRowsToContents()
    if "标题" in view.columns:
        table.setColumnWidth(list(view.columns).index("标题"), 260)
    if "类型" in view.columns:
        table.setColumnWidth(list(view.columns).index("类型"), 220)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.horizontalHeader().setStretchLastSection(True)


def poster_pixmap(title: str, genres: str, width: int = 76, height: int = 104) -> QPixmap:
    """Create a deterministic local poster thumbnail from movie metadata."""
    palettes = [
        ("#f1f1f1", "#2563EB"),
        ("#eeeeee", "#606060"),
        ("#f8f8f8", "#ff0033"),
        ("#f5f5f5", "#2563EB"),
        ("#eeeeee", "#0f0f0f"),
        ("#ffffff", "#2563EB"),
    ]
    base, accent = palettes[abs(hash(title)) % len(palettes)]
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(base))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.fillRect(0, 0, width, 10, QColor(accent))
    painter.setPen(QColor("#0f0f0f"))
    title_font = QFont("Arial", 8, QFont.Bold)
    painter.setFont(title_font)
    words = str(title).replace("(", "\n(").split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 12:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    painter.drawText(8, 24, width - 16, 52, Qt.AlignLeft | Qt.TextWordWrap, "\n".join(lines[:3]))
    painter.setPen(QColor("#606060"))
    painter.setFont(QFont("Arial", 7))
    painter.drawText(8, height - 26, width - 16, 20, Qt.AlignLeft | Qt.TextWordWrap, str(genres).split(",")[0])
    painter.end()
    return pixmap


def cover_pixmap(title: str, genres: str, width: int, height: int, image_index: int | None = None) -> QPixmap:
    """Load a provided local movie cover, cropped to poster size, with generated fallback."""
    if image_index is not None and LOCAL_COVER_IMAGES:
        image_path = LOCAL_COVER_IMAGES[image_index % len(LOCAL_COVER_IMAGES)]
        if image_path.exists():
            source = QPixmap(str(image_path))
            if not source.isNull():
                scaled = source.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                x = max(0, (scaled.width() - width) // 2)
                y = max(0, (scaled.height() - height) // 2)
                return scaled.copy(x, y, width, height)
    return poster_pixmap(title, genres, width, height)


def set_recommendation_table(table: QTableWidget, data: pd.DataFrame, max_rows: int = 25) -> None:
    """Fill recommendation tables with poster thumbnails and wrapped explanations."""
    raw_view = data.head(max_rows).copy()
    view = translate_columns(raw_view)
    columns = ["海报", *view.columns.tolist()]
    table.clear()
    table.setRowCount(len(view))
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(columns)
    raw_records = raw_view.to_dict("records")
    view_records = view.to_dict("records")
    for row_idx, (raw_row, row) in enumerate(zip(raw_records, view_records, strict=True)):
        poster = QLabel()
        poster.setPixmap(poster_pixmap(str(raw_row.get("Title", "")), str(raw_row.get("Genres", ""))))
        poster.setAlignment(Qt.AlignCenter)
        table.setCellWidget(row_idx, 0, poster)
        for col_idx, col_name in enumerate(view.columns, start=1):
            value = row.get(col_name, "")
            item = QTableWidgetItem("" if pd.isna(value) else str(value))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            item.setToolTip("" if pd.isna(value) else str(value))
            table.setItem(row_idx, col_idx, item)
        table.setRowHeight(row_idx, 112)
    table.setWordWrap(True)
    table.setColumnWidth(0, 88)
    table.resizeColumnsToContents()
    if "推荐理由" in view.columns:
        table.setColumnWidth(columns.index("推荐理由"), 380)
    if "标题" in view.columns:
        table.setColumnWidth(columns.index("标题"), 240)


def create_sidebar_stack(items: list[tuple[str, QWidget]]) -> tuple[QWidget, QListWidget, QStackedWidget]:
    """Create a YouTube-style left navigation list with readable labels."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(24)
    nav = QListWidget()
    nav.setObjectName("sideNav")
    nav.setFixedWidth(220)
    nav.setSpacing(4)
    stack = QStackedWidget()
    for label, widget in items:
        item = QListWidgetItem(label)
        item.setSizeHint(item.sizeHint())
        nav.addItem(item)
        stack.addWidget(widget)
    nav.currentRowChanged.connect(stack.setCurrentIndex)
    nav.setCurrentRow(0)
    layout.addWidget(nav)
    layout.addWidget(stack, 1)
    return container, nav, stack


def metric_label(title: str, value: str) -> QLabel:
    label = QLabel(
        f"<span style='font-size:32px; font-weight:700; color:#0f0f0f'>{value}</span>"
        f"<br><span style='font-size:12px; color:#606060'>{title}</span>"
    )
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumHeight(62)
    label.setStyleSheet(
        "QLabel { background:#ffffff; color:#0f0f0f; border:1px solid #eeeeee; "
        "border-radius:10px; padding:16px; }"
    )
    return label


class ChartWidget(QWidget):
    """Small matplotlib canvas wrapper used by both entrances."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def _style_axis(self, ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor("#ffffff")
        ax.set_title(title, color="#0f0f0f", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, color="#606060", fontsize=11)
        ax.set_ylabel(ylabel, color="#606060", fontsize=11)
        ax.tick_params(colors="#606060", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#c6c6c6")
        ax.grid(True, color="#c6c6c6", alpha=0.25)

    def bar(self, values: pd.Series, title: str, xlabel: str = "", ylabel: str = "") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        values.plot(kind="bar", ax=ax, color="#065fd4")
        self._style_axis(ax, title, xlabel, ylabel)
        self.figure.tight_layout()
        self.canvas.draw()

    def line(self, values: pd.Series, title: str, xlabel: str = "", ylabel: str = "") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        values.plot(kind="line", ax=ax, color="#065fd4", marker="o")
        self._style_axis(ax, title, xlabel, ylabel)
        self.figure.tight_layout()
        self.canvas.draw()


@dataclass
class ModelSettings:
    algorithm: str = "Item-based CF"
    k: int = 20
    metric: str = "cosine"
    factors: int = 50


class MovieLensBackend:
    """Backend facade called by the Qt interface."""

    def __init__(self) -> None:
        ratings = clean_ratings(load_ratings())
        raw_movies = clean_movies(load_movies())
        db.init_db(movies_df=raw_movies, ratings_df=ratings)
        movies = self.load_persisted_movies(raw_movies)
        self.ratings, _aligned_movies = align_ratings_with_movies(ratings, movies)
        self.movies = movies.reset_index(drop=True)
        self.avg_ratings = self.ratings.groupby("movie_id")["rating"].mean()
        self.rating_counts = self.ratings.groupby("movie_id")["rating"].count()
        self.model_cache: dict[tuple[str, int, str, int], object] = {}
        self.catalog = self._build_catalog()

    def load_persisted_movies(self, default_movies: pd.DataFrame) -> pd.DataFrame:
        movies = db.get_movies_df()
        return movies if not movies.empty else default_movies.copy()

    def refresh_movies(self) -> None:
        """Reload the movie catalog from the database after an admin edit."""
        movies = db.get_movies_df()
        self.ratings, _aligned_movies = align_ratings_with_movies(self.ratings, movies)
        self.movies = movies.reset_index(drop=True)
        self.catalog = self._build_catalog()

    def genre_columns(self) -> list[str]:
        return recsys.genre_columns(self.movies)

    def genre_text(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movie_id"] == movie_id].iloc[0]
        return recsys.genre_text(row, self.genre_columns())

    def _build_catalog(self) -> pd.DataFrame:
        rows = []
        for movie in self.movies.itertuples(index=False):
            movie_id = int(movie.movie_id)
            rows.append(
                {
                    "Movie ID": movie_id,
                    "Title": movie.title,
                    "Genres": self.genre_text(movie_id),
                    "Average Rating": round(float(self.avg_ratings.get(movie_id, np.nan)), 2),
                    "Ratings": int(self.rating_counts.get(movie_id, 0)),
                    "Release Year": self.movies.loc[
                        self.movies["movie_id"] == movie_id, "release_year"
                    ].iloc[0],
                }
            )
        return pd.DataFrame(rows)

    def dataset_stats(self) -> dict[str, str]:
        n_users = self.ratings["user_id"].nunique()
        n_movies = self.movies["movie_id"].nunique()
        sparsity = 1 - len(self.ratings) / (n_users * n_movies)
        return {
            "用户数": f"{n_users:,}",
            "电影数": f"{n_movies:,}",
            "评分数": f"{len(self.ratings):,}",
            "稀疏度": f"{sparsity * 100:.1f}%",
        }

    def user_stats(self, user_id: int) -> dict[str, str]:
        rows = self.ratings[self.ratings["user_id"] == user_id]
        latest = pd.to_datetime(rows["timestamp"].max(), unit="s").strftime("%Y-%m-%d")
        return {
            "已评分电影数": f"{len(rows):,}",
            "平均评分": f"{rows['rating'].mean():.2f}",
            "高分评价数": f"{int((rows['rating'] >= 4).sum()):,}",
            "最近活动": latest,
        }

    def user_profile(self, user_id: int) -> dict[str, str]:
        prefs = self.user_genre_preferences(user_id)
        history = self.ratings[self.ratings["user_id"] == user_id].copy()
        history["year"] = pd.to_datetime(history["timestamp"], unit="s").dt.year
        activity = len(history)
        favorite = prefs.iloc[0]["Genre"] if not prefs.empty else "未知"
        second = prefs.iloc[1]["Genre"] if len(prefs) > 1 else "未知"
        favorite_share = float(prefs.iloc[0]["Rated Movies"]) / max(1, activity) if not prefs.empty else 0.0
        if activity >= 300:
            level = "非常高"
            cluster = "电影爱好者"
        elif activity < 50:
            level = "较低"
            cluster = "轻度观影用户"
        elif favorite_share >= 0.55:
            level = "专一"
            cluster = f"{favorite} 专属爱好者"
        elif favorite == "Drama":
            level = "中等"
            cluster = "剧情片爱好者"
        elif favorite in {"Action", "Adventure", "Thriller"}:
            level = "中等"
            cluster = "动作片爱好者"
        elif favorite in {"Comedy", "Romance"}:
            level = "中等"
            cluster = "轻松娱乐型观众"
        else:
            level = "中等"
            cluster = "常规观众"
        most_active_year = int(history["year"].mode().iloc[0]) if not history.empty else 0
        return {
            "最喜欢的类型": str(favorite),
            "次喜欢的类型": str(second),
            "活跃程度": level,
            "最活跃年份": str(most_active_year),
            "用户画像": cluster,
        }

    def user_history(self, user_id: int) -> pd.DataFrame:
        history = self.ratings[self.ratings["user_id"] == user_id].merge(
            self.movies[["movie_id", "title", "release_year"]],
            on="movie_id",
            how="left",
        )
        history = history.sort_values("timestamp", ascending=False).copy()
        history["Rated At"] = pd.to_datetime(history["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        return history.rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Title",
                "release_year": "Release Year",
                "rating": "Rating",
            }
        )[["Movie ID", "Title", "Release Year", "Rating", "Rated At"]]

    def user_genre_preferences(self, user_id: int) -> pd.DataFrame:
        history = self.ratings[self.ratings["user_id"] == user_id].merge(
            self.movies[["movie_id", *self.genre_columns()]],
            on="movie_id",
            how="left",
        )
        rows = []
        for genre in self.genre_columns():
            genre_rows = history[history[genre] == 1]
            if not genre_rows.empty:
                rows.append(
                    {
                        "Genre": genre,
                        "Rated Movies": len(genre_rows),
                        "Average Rating": round(float(genre_rows["rating"].mean()), 2),
                        "Preference Score": round(
                            float(genre_rows["rating"].mean()) * np.log1p(len(genre_rows)),
                            2,
                        ),
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["Preference Score", "Rated Movies"], ascending=False
        )

    def search_movies(self, query: str, genre: str, sort_by: str = "Relevance") -> pd.DataFrame:
        result = self.catalog.copy()
        if query.strip():
            key = query.strip().lower()
            if key.isdigit():
                result = result[result["Movie ID"] == int(key)]
                result["Relevance"] = 1
            else:
                result = result[result["Title"].str.lower().str.contains(key, na=False)]
                result["Relevance"] = result["Title"].str.lower().str.contains(key, na=False).astype(int)
        else:
            result["Relevance"] = 0
        if genre != "All":
            result = result[result["Genres"].str.contains(genre, case=False, na=False)]
        sort_map = {
            "Relevance": ["Relevance", "Average Rating", "Ratings"],
            "Average Rating": ["Average Rating", "Ratings"],
            "Popularity": ["Ratings", "Average Rating"],
            "Release Year": ["Release Year", "Ratings"],
            "Number of Ratings": ["Ratings", "Average Rating"],
        }
        return result.sort_values(sort_map.get(sort_by, ["Relevance"]), ascending=False).drop(
            columns=["Relevance"],
            errors="ignore",
        )

    def movie_detail_text(self, movie_id: int) -> str:
        row = self.catalog.loc[self.catalog["Movie ID"] == movie_id].iloc[0]
        return (
            f"电影详情\n\n"
            f"标题：{row['Title']}\n"
            f"电影ID：{row['Movie ID']}\n"
            f"类型：{row['Genres']}\n"
            f"上映年份：{row['Release Year']}\n"
            f"平均评分：{row['Average Rating']}\n"
            f"评分数：{row['Ratings']}\n\n"
            "海报：本地生成的占位缩略图；MovieLens 100K 数据集不包含海报图片文件。"
        )

    def popular_movies(self, rank_by: str, top_n: int) -> pd.DataFrame:
        return recsys.popular_movies(self.catalog, rank_by, top_n, min_ratings_for_avg=50)

    def homepage_recommendation_preview(self, user_id: int = 2, top_n: int = 5) -> pd.DataFrame:
        """Fast homepage preview that avoids training a full model at startup."""
        seen = set(self.ratings[self.ratings["user_id"] == user_id]["movie_id"])
        prefs = self.user_genre_preferences(user_id).head(3)
        preferred = set(prefs["Genre"].tolist()) if not prefs.empty else set()
        ranked = self.popular_movies("Weighted Score", len(self.catalog)).copy()
        ranked = ranked[~ranked["Movie ID"].isin(seen)].copy()
        ranked["Genre Match"] = ranked["Genres"].apply(
            lambda text: len(set(str(text).split(", ")) & preferred)
        )
        ranked["Preview Score"] = (
            ranked["Weighted Score"].fillna(0) + 0.08 * ranked["Genre Match"]
        ).round(2)
        ranked = ranked.sort_values(["Preview Score", "Ratings"], ascending=False).head(top_n)
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        return ranked[["Rank", "Title", "Preview Score", "Genres"]]

    def model(self, settings: ModelSettings) -> object:
        key = (settings.algorithm, settings.k, settings.metric, settings.factors)
        if key in self.model_cache:
            return self.model_cache[key]
        model = recsys.build_model(
            self.ratings,
            settings.algorithm,
            k=settings.k,
            metric=settings.metric,
            factors=settings.factors,
        )
        self.model_cache[key] = model
        return model

    def recommendations(self, user_id: int, settings: ModelSettings, top_n: int) -> pd.DataFrame:
        model = self.model(settings)
        seen = set(self.ratings[self.ratings["user_id"] == user_id]["movie_id"])
        user_mean = float(self.ratings.loc[self.ratings["user_id"] == user_id, "rating"].mean())
        global_mean = float(self.ratings["rating"].mean())
        rows = []
        for movie_id in self.movies["movie_id"]:
            movie_id = int(movie_id)
            if movie_id in seen:
                continue
            raw_pred = float(model.predict(user_id, movie_id))
            avg_rating = float(self.avg_ratings.get(movie_id, global_mean))
            confidence, support_count, avg_similarity = self.recommendation_confidence(
                model, settings, user_id, movie_id
            )
            calibrated = self.calibrated_score(
                raw_pred,
                avg_rating,
                user_mean,
                confidence,
                int(self.rating_counts.get(movie_id, 0)),
            )
            genres = self.genre_text(movie_id)
            reason = recsys.recommendation_reason(settings.algorithm, genres)
            rows.append(
                {
                    "Movie ID": movie_id,
                    "Title": self.movies.loc[self.movies["movie_id"] == movie_id, "title"].iloc[0],
                    "Recommendation Score": calibrated,
                    "Raw Model Score": raw_pred,
                    "Average Rating": avg_rating,
                    "Confidence": confidence,
                    "Neighbor Count": support_count,
                    "Average Similarity": avg_similarity,
                    "Genres": genres,
                    "Reason": reason,
                }
            )
        recs = pd.DataFrame(rows).sort_values(
            ["Recommendation Score", "Confidence", "Average Rating"], ascending=False
        ).head(top_n)
        recs.insert(0, "Rank", range(1, len(recs) + 1))
        recs["Recommendation Score"] = recs["Recommendation Score"].round(2)
        recs["Raw Model Score"] = recs["Raw Model Score"].round(2)
        recs["Average Rating"] = recs["Average Rating"].round(2)
        recs["Confidence"] = recs["Confidence"].round(2)
        recs["Average Similarity"] = recs["Average Similarity"].round(3)
        return recs

    def calibrated_score(
        self,
        raw_pred: float,
        avg_rating: float,
        user_mean: float,
        confidence: float,
        rating_count: int,
    ) -> float:
        """Shrink clipped model outputs toward reliable evidence for display/ranking."""
        return recsys.calibrated_score(
            raw_pred,
            avg_rating,
            user_mean,
            confidence,
            rating_count,
            float(self.ratings["rating"].mean()),
        )

    def recommendation_confidence(
        self,
        model: object,
        settings: ModelSettings,
        user_id: int,
        movie_id: int,
    ) -> tuple[float, int, float]:
        """Estimate support strength for a recommendation."""
        support_count = int(self.rating_counts.get(movie_id, 0))
        avg_similarity = 0.0
        if settings.algorithm == "Item-based CF" and getattr(model, "_sim", None) is not None:
            matrix = getattr(model, "_matrix", None)
            sim = getattr(model, "_sim", None)
            if matrix is not None and user_id in matrix.index and movie_id in sim.index:
                rated_items = matrix.loc[user_id].dropna().index
                sims = sim.loc[movie_id, rated_items].drop(labels=[movie_id], errors="ignore")
                sims = sims[sims > 0].sort_values(ascending=False).head(settings.k)
                support_count = int(len(sims))
                avg_similarity = float(sims.mean()) if len(sims) else 0.0
        elif settings.algorithm == "User-based CF" and getattr(model, "_sim", None) is not None:
            matrix = getattr(model, "_matrix", None)
            sim = getattr(model, "_sim", None)
            if matrix is not None and user_id in sim.index and movie_id in matrix.columns:
                raters = matrix[movie_id].dropna().index
                sims = sim.loc[user_id, raters].drop(labels=[user_id], errors="ignore")
                sims = sims[sims > 0].sort_values(ascending=False).head(settings.k)
                support_count = int(len(sims))
                avg_similarity = float(sims.mean()) if len(sims) else 0.0
        else:
            support_count = int(self.rating_counts.get(movie_id, 0))
            avg_similarity = min(1.0, np.log1p(support_count) / np.log1p(250))

        support_component = min(1.0, support_count / max(1, settings.k))
        similarity_component = min(1.0, max(0.0, avg_similarity))
        popularity_component = min(1.0, np.log1p(int(self.rating_counts.get(movie_id, 0))) / np.log1p(250))
        confidence = 0.45 * support_component + 0.35 * similarity_component + 0.20 * popularity_component
        return float(confidence), support_count, float(avg_similarity)

    def similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        model = self.model(ModelSettings("Item-based CF", 20, "cosine", 50))
        similar = model.similar_items(movie_id, n=top_n)
        rows = []
        for rank, (other_id, score) in enumerate(similar.items(), 1):
            rows.append(
                {
                    "Rank": rank,
                    "Movie ID": int(other_id),
                    "Title": self.movies.loc[self.movies["movie_id"] == other_id, "title"].iloc[0],
                    "Similarity": round(float(score), 4),
                    "Genres": self.genre_text(int(other_id)),
                    "Average Rating": round(float(self.avg_ratings.get(other_id, np.nan)), 2),
                }
            )
        return pd.DataFrame(rows)

    def topn_evaluation(self) -> pd.DataFrame:
        train, test = user_temporal_split(self.ratings)
        users = sorted(test["user_id"].unique())[:60]
        all_items = set(train["movie_id"]) | set(test["movie_id"])
        relevant = relevant_items_by_user(test[test["user_id"].isin(users)], threshold=4.0)
        candidates = sampled_candidate_items_by_user(
            train,
            test,
            users,
            all_items,
            threshold=4.0,
            n_negatives=100,
            random_state=42,
        )
        models = {
            "User-based CF": UserBasedCF(k=20).fit(train),
            "Item-based CF": ItemBasedCF(k=20).fit(train),
            "SVD": SVDRecommender(n_components=50).fit(train),
        }
        rows = []
        for name, model in models.items():
            recs = recommend_top_k_from_model(
                model,
                train,
                users,
                all_items,
                k=10,
                max_users=None,
                candidate_items_by_user=candidates,
            )
            metrics = evaluate_topk(recs, relevant, k=10)
            rows.append(
                {
                    "Algorithm": name,
                    "Precision@10": round(metrics["precision_at_k"], 4),
                    "Recall@10": round(metrics["recall_at_k"], 4),
                    "HitRate@10": round(metrics["hit_rate_at_k"], 4),
                    "NDCG@10": round(metrics["ndcg_at_k"], 4),
                }
            )
        return pd.DataFrame(rows)

    def significance_summary(self) -> pd.DataFrame:
        train, test = user_temporal_split(self.ratings)
        sample = test.sample(n=min(1000, len(test)), random_state=42)
        y_true = sample["rating"].to_numpy(dtype=float)
        item_pred = ItemBasedCF(k=20).fit(train).predict_batch(sample)
        svd_pred = SVDRecommender(n_components=50).fit(train).predict_batch(sample)
        result = paired_error_test(y_true, item_pred, svd_pred)
        return pd.DataFrame(
            [
                {
                    "Comparison": "ItemCF vs SVD",
                    "MSE Difference": round(result["mean_squared_error_diff"], 4),
                    "t-stat": round(result["t_stat"], 4),
                    "p-value": round(result["p_value"], 6),
                    "Effect Size": round(result["cohens_d_paired"], 4),
                }
            ]
        )

    def evaluate_models(self) -> pd.DataFrame:
        train, test = user_temporal_split(self.ratings)
        sample = test.sample(n=min(1500, len(test)), random_state=42)
        models = {
            "User-based CF": UserBasedCF(k=20).fit(train),
            "Item-based CF": ItemBasedCF(k=20).fit(train),
            "SVD": SVDRecommender(n_components=50).fit(train),
        }
        y_true = sample["rating"].to_numpy(dtype=float)
        rows = []
        for name, model in models.items():
            pred = model.predict_batch(sample)
            rows.append(
                {
                    "Algorithm": name,
                    "MAE": round(mae(y_true, pred), 4),
                    "RMSE": round(rmse(y_true, pred), 4),
                    "Evaluated Ratings": len(sample),
                }
            )
        return pd.DataFrame(rows)

    def svd_factor_experiment(self) -> pd.DataFrame:
        train, test = user_temporal_split(self.ratings)
        sample = test.sample(n=min(1200, len(test)), random_state=42)
        y_true = sample["rating"].to_numpy(dtype=float)
        rows = []
        for factors in [10, 20, 50, 80]:
            model = SVDRecommender(n_components=factors).fit(train)
            pred = model.predict_batch(sample)
            rows.append(
                {
                    "Factors": factors,
                    "MAE": round(mae(y_true, pred), 4),
                    "RMSE": round(rmse(y_true, pred), 4),
                }
            )
        return pd.DataFrame(rows)

    def coverage_and_diversity(self) -> dict[str, str]:
        sample_users = list(map(int, self.ratings["user_id"].drop_duplicates().head(3)))
        settings = ModelSettings("SVD", 20, "cosine", 50)
        recommended: set[int] = set()
        diversity_scores = []
        genre_cols = self.genre_columns()
        for user_id in sample_users:
            recs = self.recommendations(user_id, settings, 10)
            ids = [int(movie_id) for movie_id in recs["Movie ID"]]
            recommended.update(ids)
            genres = set()
            for movie_id in ids:
                genres.update(self.genre_text(movie_id).split(", "))
            diversity_scores.append(len(genres) / max(1, len(genre_cols)))
        coverage = len(recommended) / self.movies["movie_id"].nunique()
        return {
            "Top-N 覆盖率": f"{coverage * 100:.1f}%",
            "平均多样性": f"{np.mean(diversity_scores):.2f}",
            "已推荐电影数": f"{len(recommended):,}",
        }

    def recommendation_statistics(self) -> pd.DataFrame:
        eval_df = self.evaluate_models()
        best = eval_df.sort_values("RMSE").iloc[0]
        quality = self.coverage_and_diversity()
        rows = eval_df[["Algorithm", "RMSE", "MAE"]].copy()
        rows["Best Algorithm"] = ""
        rows.loc[rows["Algorithm"] == best["Algorithm"], "Best Algorithm"] = "是"
        rows["Top-N Coverage"] = quality["Top-N 覆盖率"]
        rows["Avg Diversity"] = quality["平均多样性"]
        return rows


class WelcomePage(QWidget):
    def __init__(self, main_window: MainWindow) -> None:
        super().__init__()
        self.backend = main_window.backend
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 24, 32, 40)
        layout.setSpacing(24)

        hero = QWidget()
        hero.setStyleSheet("background:#ffffff; border:1px solid #e5e5e5; border-radius:16px;")
        hero.setMaximumHeight(250)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(32, 24, 32, 24)
        hero_layout.setSpacing(32)
        hero_text = QVBoxLayout()
        kicker = QLabel("FilmTrace · MovieLens 推荐系统")
        kicker.setStyleSheet("QLabel { color:#2563EB; font-size:13px; font-weight:600; border:0; background:transparent; }")
        title = QLabel("发现你的下一部最爱电影")
        title.setWordWrap(True)
        title.setStyleSheet("QLabel { font-size:34px; font-weight:700; color:#0f0f0f; border:0; background:transparent; }")
        subtitle = QLabel("基于 MovieLens 100K 数据集，提供 UserCF、ItemCF 和 SVD 推荐模型。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("QLabel { font-size:15px; color:#606060; border:0; background:transparent; }")
        button_row = QHBoxLayout()
        user_button = QPushButton("浏览电影")
        user_button.setProperty("class", "primary")
        user_button.setStyleSheet("QPushButton { background:#2563EB; color:#ffffff; border-radius:18px; padding:9px 20px; font-weight:600; }")
        admin_button = QPushButton("打开管理后台")
        admin_button.setProperty("class", "secondary")
        admin_button.setStyleSheet("QPushButton { background:#F3F4F6; color:#0f0f0f; border-radius:18px; padding:9px 20px; font-weight:600; }")
        user_button.clicked.connect(lambda: main_window.show_page("user"))
        admin_button.clicked.connect(lambda: main_window.show_page("admin"))
        button_row.addWidget(user_button)
        button_row.addWidget(admin_button)
        button_row.addStretch()
        hero_text.addWidget(kicker)
        hero_text.addWidget(title)
        hero_text.addWidget(subtitle)
        hero_text.addSpacing(8)
        hero_text.addLayout(button_row)
        featured = self.featured_movie_card(image_index=1)
        hero_layout.addLayout(hero_text, 2)
        hero_layout.addWidget(featured, 1)
        layout.addWidget(hero)

        stats_row = QHBoxLayout()
        for title_text, value in self.backend.dataset_stats().items():
            stats_row.addWidget(self.info_card(value, title_text, self.stat_caption(title_text)))
        layout.addLayout(stats_row)

        layout.addWidget(self.section_title("热门电影"))
        trending = QHBoxLayout()
        for movie in self.backend.popular_movies("Weighted Score", 6).to_dict("records"):
            trending.addWidget(self.movie_card(movie["Title"], movie["Genres"], movie["Average Rating"]))
        layout.addLayout(trending)

        layout.addWidget(self.section_title("系统能力展示"))
        tech_row = QHBoxLayout()
        tech_cards = [
            ("用户匹配", "User-based CF", "根据兴趣相似的用户推荐电影。"),
            ("影片匹配", "Item-based CF", "推荐与你已喜欢电影相似的影片。"),
            ("潜在偏好", "SVD", "通过隐因子学习用户的潜在兴趣。"),
        ]
        for icon, name, desc in tech_cards:
            tech_row.addWidget(self.text_card(name, desc, icon))
        layout.addLayout(tech_row)

        flow_row = QHBoxLayout()
        flow_row.addWidget(
            self.text_card(
                "推荐流程",
                "MovieLens 数据集\n↓\nUserCF / ItemCF / SVD\n↓\n模型评估\n↓\nTop-N 推荐结果",
                "流程",
            )
        )
        preview = self.backend.homepage_recommendation_preview(2, 5)
        preview_text = "\n".join(
            f"{int(row['Rank'])}. {row['Title']}    {float(row['Preview Score']):.2f}"
            for row in preview.to_dict("records")
        )
        flow_row.addWidget(self.text_card("推荐结果预览（用户 #2）", preview_text, "预览"))
        layout.addLayout(flow_row)

        chart_row = QHBoxLayout()
        rating_chart = ChartWidget()
        rating_chart.bar(self.backend.ratings["rating"].value_counts().sort_index(), "评分分布", "评分", "数量")
        genre_chart = ChartWidget()
        genre_counts = pd.Series(
            {genre: int(self.backend.movies[genre].sum()) for genre in self.backend.genre_columns()}
        ).sort_values(ascending=False).head(10)
        genre_chart.bar(genre_counts, "热门类型", "类型", "电影数")
        chart_row.addWidget(rating_chart)
        chart_row.addWidget(genre_chart)
        layout.addLayout(chart_row)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size:20px; font-weight:600; color:#0f0f0f;")
        return label

    def stat_caption(self, title: str) -> str:
        captions = {
            "用户数": "用户总数",
            "电影数": "MovieLens 电影库规模",
            "评分数": "已观测到的评分记录数",
            "稀疏度": "缺失的用户-电影评分占比",
        }
        return captions.get(title, "数据集指标")

    def info_card(self, value: str, title: str, caption: str) -> QLabel:
        icons = {"用户数": "用户", "电影数": "电影", "评分数": "评分", "稀疏度": "矩阵"}
        card = QLabel(
            f"<span style='font-size:12px; color:#2563EB; font-weight:700'>{icons.get(title, '数据')}</span><br>"
            f"<span style='font-size:30px; font-weight:700; color:#0f0f0f'>{value}</span><br>"
            f"<span style='font-size:14px; font-weight:600'>{title}</span><br>"
            f"<span style='font-size:12px; color:#606060'>{caption}</span>"
        )
        card.setMinimumHeight(118)
        card.setAlignment(Qt.AlignCenter)
        card.setStyleSheet("background:#ffffff; border:1px solid #e5e5e5; border-radius:12px; padding:16px;")
        return card

    def text_card(self, title: str, body: str, icon: str = "") -> QLabel:
        card = QLabel(
            f"<span style='color:#2563EB; font-size:12px; font-weight:700'>{icon}</span><br>"
            f"<span style='font-size:16px; font-weight:700'>{title}</span><br>"
            f"<span style='color:#606060; line-height:1.35'>{body}</span>"
        )
        card.setWordWrap(True)
        card.setMinimumHeight(126)
        card.setStyleSheet("QLabel { background:#ffffff; border:1px solid #e5e5e5; border-radius:12px; padding:18px; }")
        return card

    def featured_movie_card(self, image_index: int | None = None) -> QWidget:
        card = QWidget()
        card.setStyleSheet("QWidget { background:#F8FAFC; border:1px solid #e5e5e5; border-radius:14px; }")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)
        poster_strip = QHBoxLayout()
        poster_strip.setSpacing(8)
        for idx, (name, genre) in enumerate([("Inside Out", "Animation"), ("Avatar", "Sci-Fi"), ("Zootopia", "Animation")]):
            poster = QLabel()
            poster.setAlignment(Qt.AlignCenter)
            poster.setPixmap(cover_pixmap(name, genre, 72, 104, idx))
            poster.setStyleSheet("QLabel { border:0; background:transparent; }")
            poster_strip.addWidget(poster)
        detail = QLabel(
            "<span style='font-size:12px; color:#2563EB; font-weight:700'>海报展示</span><br>"
            "<span style='font-size:18px; font-weight:700'>电影发现，从这里开始</span><br>"
            "<span style='color:#606060'>真实的封面图能够提升展示效果，让推荐系统更具产品质感。</span><br>"
            "<span style='font-weight:700'>前往“为你推荐”页面查看完整 Top-N 推荐结果</span>"
        )
        detail.setWordWrap(True)
        detail.setStyleSheet("QLabel { border:0; background:transparent; color:#0f0f0f; }")
        layout.addLayout(poster_strip)
        layout.addWidget(detail, 1)
        return card

    def movie_card(
        self,
        title: str,
        genres: str,
        rating: float,
        compact: bool = False,
        image_index: int | None = None,
    ) -> QWidget:
        card = QWidget()
        card.setStyleSheet("QWidget { background:#ffffff; border:1px solid #e5e5e5; border-radius:12px; }")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        poster = QLabel()
        poster.setAlignment(Qt.AlignCenter)
        poster.setPixmap(cover_pixmap(str(title), str(genres), 132, 186, image_index))
        poster.setStyleSheet("QLabel { border:0; background:transparent; }")
        meta = QLabel(f"<b>评分 {float(rating):.2f}</b><br><span style='color:#606060'>{str(genres).split(',')[0]}</span>")
        meta.setStyleSheet("QLabel { color:#0f0f0f; font-size:12px; border:0; background:transparent; }")
        layout.addWidget(poster)
        layout.addWidget(meta)
        card.setMinimumWidth(170)
        return card


class MovieDetailDialog(QDialog):
    def __init__(self, backend: MovieLensBackend, movie_id: int, reason: str = "") -> None:
        super().__init__()
        self.setWindowTitle("电影详情")
        self.resize(620, 420)
        movie = backend.catalog.loc[backend.catalog["Movie ID"] == movie_id].iloc[0]
        layout = QHBoxLayout(self)
        poster = QLabel()
        poster.setPixmap(poster_pixmap(str(movie["Title"]), str(movie["Genres"]), 150, 210))
        poster.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        detail = QTextEdit()
        detail.setReadOnly(True)
        similar = backend.similar_movies(movie_id, top_n=5)
        similar_text = "\n".join(f"- {row.Title}" for row in similar.itertuples(index=False))
        detail.setText(
            f"标题：{movie['Title']}\n"
            f"类型：{movie['Genres']}\n"
            f"上映年份：{movie['Release Year']}\n"
            f"平均评分：{movie['Average Rating']}\n"
            f"评分数：{movie['Ratings']}\n\n"
            f"推荐理由：\n{reason or '从电影列表打开。'}\n\n"
            f"相似电影：\n{similar_text}\n\n"
            "说明：由于 MovieLens 100K 数据集不提供海报图片文件，"
            "此处显示的是本地生成的占位缩略图。"
        )
        layout.addWidget(poster)
        layout.addWidget(detail)


class UserPage(QWidget):
    def __init__(self, backend: MovieLensBackend) -> None:
        super().__init__()
        self.backend = backend
        self.current_user = int(backend.ratings["user_id"].min())
        self.settings = ModelSettings()
        layout = QVBoxLayout(self)

        login_box = QGroupBox("用户登录 / 选择")
        login_layout = QGridLayout(login_box)
        self.user_spin = QSpinBox()
        self.user_spin.setRange(int(backend.ratings["user_id"].min()), int(backend.ratings["user_id"].max()))
        self.user_spin.setValue(self.current_user)
        load_btn = QPushButton("加载用户")
        load_btn.setProperty("class", "secondary")
        load_btn.clicked.connect(self.load_user)
        login_layout.addWidget(QLabel(T["label_user_id"]), 0, 0)
        login_layout.addWidget(self.user_spin, 0, 1)
        login_layout.addWidget(load_btn, 0, 2)
        self.stats_layout = QHBoxLayout()
        login_layout.addLayout(self.stats_layout, 1, 0, 1, 3)
        self.profile_layout = QHBoxLayout()
        login_layout.addLayout(self.profile_layout, 2, 0, 1, 3)
        layout.addWidget(login_box)

        self.recommendation_tab = QWidget()
        self.search_tab = QWidget()
        self.history_tab = QWidget()
        self.popular_tab = QWidget()
        self.visual_tab = QWidget()
        self.similar_tab = QWidget()
        user_container, self.user_nav, self.user_stack = create_sidebar_stack(
            [
                (f"🎯 {T['nav_for_you']}", self.recommendation_tab),
                (f"🔎 {T['nav_catalog']}", self.search_tab),
                (f"🕘 {T['nav_rating_records']}", self.history_tab),
                ("🔥 热门电影", self.popular_tab),
                (f"📊 {T['nav_visualization']}", self.visual_tab),
                (f"🎬 {T['nav_similar_movies']}", self.similar_tab),
            ]
        )
        layout.addWidget(user_container, 1)

        self._build_recommendation_tab()
        self._build_search_tab()
        self._build_history_tab()
        self._build_popular_tab()
        self._build_visual_tab()
        self._build_similar_tab()
        self.load_user()

    def load_user(self) -> None:
        self.current_user = int(self.user_spin.value())
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for title, value in self.backend.user_stats(self.current_user).items():
            self.stats_layout.addWidget(metric_label(title, value))
        while self.profile_layout.count():
            item = self.profile_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for title, value in self.backend.user_profile(self.current_user).items():
            self.profile_layout.addWidget(metric_label(title, value))
        self.refresh_history()

    def _build_recommendation_tab(self) -> None:
        layout = QVBoxLayout(self.recommendation_tab)
        controls = QHBoxLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Item-based CF", "User-based CF", "SVD"])
        self.algorithm_combo.currentTextChanged.connect(self.update_parameter_visibility)
        self.topn_combo = QComboBox()
        self.topn_combo.addItems(["5", "10", "20"])
        self.k_spin = QSpinBox()
        self.k_spin.setRange(5, 50)
        self.k_spin.setValue(20)
        self.factor_spin = QSpinBox()
        self.factor_spin.setRange(10, 80)
        self.factor_spin.setValue(50)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["cosine", "pearson"])
        run_btn = QPushButton(T["btn_generate_recommendations"])
        run_btn.setProperty("class", "primary")
        run_btn.clicked.connect(self.refresh_recommendations)
        self.algorithm_label = QLabel(T["label_algorithm"])
        self.topn_label = QLabel("Top N")
        self.k_label = QLabel("K")
        self.factor_label = QLabel("Factors")
        self.metric_label = QLabel(T["label_similarity_method"])
        for label_widget, widget in [
            (self.algorithm_label, self.algorithm_combo),
            (self.topn_label, self.topn_combo),
            (self.k_label, self.k_spin),
            (self.factor_label, self.factor_spin),
            (self.metric_label, self.metric_combo),
        ]:
            controls.addWidget(label_widget)
            controls.addWidget(widget)
        controls.addWidget(run_btn)
        self.rec_table = QTableWidget()
        self.rec_table.cellDoubleClicked.connect(self.open_recommended_movie_detail)
        self.current_recs = pd.DataFrame()
        layout.addLayout(controls)
        layout.addWidget(self.rec_table)
        self.update_parameter_visibility()

    def update_parameter_visibility(self) -> None:
        algorithm = self.algorithm_combo.currentText()
        uses_neighbors = algorithm in {"Item-based CF", "User-based CF"}
        uses_factors = algorithm == "SVD"
        self.k_label.setVisible(uses_neighbors)
        self.k_spin.setVisible(uses_neighbors)
        self.metric_label.setVisible(uses_neighbors)
        self.metric_combo.setVisible(uses_neighbors)
        self.factor_label.setVisible(uses_factors)
        self.factor_spin.setVisible(uses_factors)

    def refresh_recommendations(self) -> None:
        self.settings = ModelSettings(
            self.algorithm_combo.currentText(),
            int(self.k_spin.value()),
            self.metric_combo.currentText(),
            int(self.factor_spin.value()),
        )
        recs = self.backend.recommendations(self.current_user, self.settings, int(self.topn_combo.currentText()))
        self.current_recs = recs
        set_recommendation_table(self.rec_table, recs, max_rows=25)

    def open_recommended_movie_detail(self, row: int, _col: int) -> None:
        if self.current_recs.empty or row >= len(self.current_recs):
            return
        rec = self.current_recs.iloc[row]
        dialog = MovieDetailDialog(self.backend, int(rec["Movie ID"]), str(rec.get("Reason", "")))
        dialog.exec_()

    def _build_search_tab(self) -> None:
        layout = QVBoxLayout(self.search_tab)
        controls = QHBoxLayout()
        self.search_input = QLineEdit()
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(["All", *self.backend.genre_columns()])
        self.search_sort_combo = QComboBox()
        self.search_sort_combo.addItems(["Relevance", "Average Rating", "Popularity", "Release Year", "Number of Ratings"])
        search_btn = QPushButton(T["btn_search"])
        search_btn.setProperty("class", "secondary")
        search_btn.clicked.connect(self.refresh_search)
        controls.addWidget(QLabel(T["label_title"] + " / " + T["label_movie_id"]))
        controls.addWidget(self.search_input)
        controls.addWidget(QLabel(T["label_genre"]))
        controls.addWidget(self.genre_combo)
        controls.addWidget(QLabel(T["label_rank_by"]))
        controls.addWidget(self.search_sort_combo)
        controls.addWidget(search_btn)
        self.search_table = QTableWidget()
        layout.addLayout(controls)
        layout.addWidget(self.search_table)
        self.refresh_search()

    def refresh_search(self) -> None:
        data = self.backend.search_movies(
            self.search_input.text(),
            self.genre_combo.currentText(),
            self.search_sort_combo.currentText(),
        )
        set_table_data(self.search_table, data, max_rows=300)

    def _build_history_tab(self) -> None:
        layout = QVBoxLayout(self.history_tab)
        self.history_table = QTableWidget()
        self.pref_table = QTableWidget()
        layout.addWidget(QLabel("该用户已评分的电影"))
        layout.addWidget(self.history_table)
        layout.addWidget(QLabel("类型偏好分析"))
        layout.addWidget(self.pref_table)

    def refresh_history(self) -> None:
        set_table_data(self.history_table, self.backend.user_history(self.current_user), max_rows=300)
        set_table_data(self.pref_table, self.backend.user_genre_preferences(self.current_user), max_rows=30)

    def _build_popular_tab(self) -> None:
        layout = QVBoxLayout(self.popular_tab)
        controls = QHBoxLayout()
        self.popular_rank_combo = QComboBox()
        self.popular_rank_combo.addItems(["Weighted Score", "Average Rating", "Number of Ratings"])
        self.popular_n_combo = QComboBox()
        self.popular_n_combo.addItems(["10", "20"])
        refresh_btn = QPushButton(T["btn_refresh_data"])
        refresh_btn.setProperty("class", "secondary")
        refresh_btn.clicked.connect(self.refresh_popular)
        controls.addWidget(QLabel(T["label_rank_by"]))
        controls.addWidget(self.popular_rank_combo)
        controls.addWidget(QLabel(T["label_show_top"]))
        controls.addWidget(self.popular_n_combo)
        controls.addWidget(refresh_btn)
        self.popular_table = QTableWidget()
        layout.addLayout(controls)
        layout.addWidget(self.popular_table)
        self.refresh_popular()

    def refresh_popular(self) -> None:
        data = self.backend.popular_movies(self.popular_rank_combo.currentText(), int(self.popular_n_combo.currentText()))
        set_table_data(self.popular_table, data, max_rows=25)

    def _build_visual_tab(self) -> None:
        layout = QGridLayout(self.visual_tab)
        self.rating_chart = ChartWidget()
        self.genre_chart = ChartWidget()
        self.activity_chart = ChartWidget()
        layout.addWidget(self.rating_chart, 0, 0)
        layout.addWidget(self.genre_chart, 0, 1)
        layout.addWidget(self.activity_chart, 1, 0, 1, 2)
        refresh_btn = QPushButton(T["btn_refresh_data"])
        refresh_btn.setProperty("class", "secondary")
        refresh_btn.clicked.connect(self.refresh_charts)
        layout.addWidget(refresh_btn, 2, 0, 1, 2)
        self.refresh_charts()

    def refresh_charts(self) -> None:
        self.rating_chart.bar(
            self.backend.ratings["rating"].value_counts().sort_index(), "评分分布", "评分", "数量"
        )
        prefs = self.backend.user_genre_preferences(self.current_user)
        self.genre_chart.bar(prefs.set_index("Genre")["Rated Movies"].head(10), "用户类型偏好", "类型", "已评分电影数")
        user_rows = self.backend.ratings[self.backend.ratings["user_id"] == self.current_user].copy()
        user_rows["date"] = pd.to_datetime(user_rows["timestamp"], unit="s").dt.date
        self.activity_chart.line(user_rows.groupby("date")["rating"].count(), "用户评分行为", "日期", "评分次数")

    def _build_similar_tab(self) -> None:
        layout = QVBoxLayout(self.similar_tab)
        controls = QHBoxLayout()
        self.similar_movie_input = QSpinBox()
        self.similar_movie_input.setRange(int(self.backend.movies["movie_id"].min()), int(self.backend.movies["movie_id"].max()))
        self.similar_movie_input.setValue(1)
        run_btn = QPushButton("查找相似电影")
        run_btn.setProperty("class", "secondary")
        run_btn.clicked.connect(self.refresh_similar)
        controls.addWidget(QLabel(T["label_movie_id"]))
        controls.addWidget(self.similar_movie_input)
        controls.addWidget(run_btn)
        method_note = QLabel(
            "相似度计算方法：基于均值中心化评分向量的物品-物品余弦相似度，"
            "依据用户评分行为计算，而非海报或剧情元数据。"
        )
        method_note.setWordWrap(True)
        self.similar_table = QTableWidget()
        layout.addLayout(controls)
        layout.addWidget(method_note)
        layout.addWidget(self.similar_table)

    def refresh_similar(self) -> None:
        data = self.backend.similar_movies(int(self.similar_movie_input.value()), top_n=10)
        set_table_data(self.similar_table, data, max_rows=20)


class AdminPage(QWidget):
    def __init__(self, backend: MovieLensBackend) -> None:
        super().__init__()
        self.backend = backend
        self.logged_in = False
        layout = QVBoxLayout(self)
        self.login_box = QGroupBox("管理员登录")
        form = QFormLayout(self.login_box)
        self.username_input = QLineEdit("admin")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        login_btn = QPushButton("登录")
        login_btn.setProperty("class", "primary")
        login_btn.clicked.connect(self.login)
        form.addRow(T["label_username"], self.username_input)
        form.addRow(T["label_password"], self.password_input)
        form.addWidget(login_btn)
        self.admin_status_box = QGroupBox("管理员状态")
        status_layout = QHBoxLayout(self.admin_status_box)
        self.admin_login_status = metric_label("当前角色", "管理员")
        self.admin_model_status = metric_label("当前模型", "Item-based CF")
        self.admin_time_status = metric_label("数据集", "MovieLens 100K")
        status_layout.addWidget(self.admin_login_status)
        status_layout.addWidget(self.admin_model_status)
        status_layout.addWidget(self.admin_time_status)
        self.admin_status_box.hide()
        self.current_admin_settings = ModelSettings()
        layout.addWidget(self.login_box)
        layout.addWidget(self.admin_status_box)

        self.dashboard_tab = QWidget()
        self.movie_tab = QWidget()
        self.user_tab = QWidget()
        self.rating_tab = QWidget()
        self.algorithm_tab = QWidget()
        self.admin_recommendation_tab = QWidget()
        self.eval_tab = QWidget()
        self.stats_tab = QWidget()
        admin_container, self.admin_nav, self.admin_stack = create_sidebar_stack(
            [
                (f"📊 {T['nav_home']}", self.dashboard_tab),
                (f"🎬 {T['nav_catalog']}", self.movie_tab),
                (f"👤 {T['nav_user_management']}", self.user_tab),
                (f"⭐ {T['nav_rating_records']}", self.rating_tab),
                (f"⚙ {T['nav_algorithm_config']}", self.algorithm_tab),
                (f"🎯 {T['nav_for_you']}", self.admin_recommendation_tab),
                (f"📈 {T['nav_model_evaluation']}", self.eval_tab),
                (f"📋 {T['nav_system_stats']}", self.stats_tab),
            ]
        )
        self.admin_content_container = admin_container
        self.admin_content_container.setEnabled(False)
        layout.addWidget(self.admin_content_container, 1)

        self._build_dashboard_tab()
        self._build_movie_tab()
        self._build_user_tab()
        self._build_rating_tab()
        self._build_algorithm_tab()
        self._build_admin_recommendation_tab()
        self._build_eval_tab()
        self._build_stats_tab()

    def login(self) -> None:
        if verify_admin(self.username_input.text(), self.password_input.text()):
            self.admin_username = self.username_input.text().strip() or "admin"
            self.admin_content_container.setEnabled(True)
            self.login_box.hide()
            self.admin_status_box.show()
            QMessageBox.information(self, "登录", T["msg_admin_verified"])
        else:
            QMessageBox.warning(self, "登录", "用户名或密码错误。")

    def _build_dashboard_tab(self) -> None:
        layout = QVBoxLayout(self.dashboard_tab)
        stats = QGridLayout()
        dashboard_stats = self.backend.dataset_stats()
        dashboard_stats["平均评分"] = f"{self.backend.ratings['rating'].mean():.2f}"
        dashboard_stats["最活跃用户"] = str(int(self.backend.ratings.groupby("user_id").size().idxmax()))
        dashboard_stats["评分最多的电影"] = str(
            self.backend.catalog.sort_values("Ratings", ascending=False).iloc[0]["Title"]
        )
        top_genre = max(
            self.backend.genre_columns(),
            key=lambda genre: int(self.backend.movies[genre].sum()),
        )
        dashboard_stats["最热门类型"] = top_genre
        for idx, (title, value) in enumerate(dashboard_stats.items()):
            stats.addWidget(metric_label(title, value), idx // 4, idx % 4)
        self.admin_rating_chart = ChartWidget()
        findings = QLabel(
            "关键发现：MovieLens 100K 数据集非常稀疏，因此协同过滤算法需要基于有限的重叠评分来推断用户偏好。"
            "评分 4 分是最常见的打分，且影片类型分布十分广泛。"
        )
        findings.setWordWrap(True)
        findings.setStyleSheet("background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; padding:16px; color:#606060;")
        layout.addLayout(stats)
        layout.addWidget(findings)
        layout.addWidget(self.admin_rating_chart)
        self.admin_rating_chart.bar(
            self.backend.ratings["rating"].value_counts().sort_index(), "评分分布", "评分", "数量"
        )

    def _build_movie_tab(self) -> None:
        layout = QVBoxLayout(self.movie_tab)
        controls = QHBoxLayout()
        self.admin_movie_search = QLineEdit()
        self.admin_movie_genre = QComboBox()
        self.admin_movie_genre.addItems(["All", *self.backend.genre_columns()])
        self.admin_movie_sort = QComboBox()
        self.admin_movie_sort.addItems(["Relevance", "Average Rating", "Popularity", "Release Year", "Number of Ratings"])
        search_btn = QPushButton(T["btn_search"])
        search_btn.setProperty("class", "secondary")
        search_btn.clicked.connect(self.refresh_admin_movies)
        controls.addWidget(QLabel(T["label_movie_id"] + " / " + T["label_title"]))
        controls.addWidget(self.admin_movie_search)
        controls.addWidget(QLabel(T["label_genre"]))
        controls.addWidget(self.admin_movie_genre)
        controls.addWidget(QLabel(T["label_rank_by"]))
        controls.addWidget(self.admin_movie_sort)
        controls.addWidget(search_btn)
        action_controls = QHBoxLayout()
        add_btn = QPushButton(T["btn_add_movie"])
        edit_btn = QPushButton(T["btn_edit_movie"])
        delete_btn = QPushButton(T["btn_delete_movie"])
        refresh_btn = QPushButton(T["btn_refresh_data"])
        for btn in [add_btn, edit_btn, refresh_btn]:
            btn.setProperty("class", "secondary")
        delete_btn.setProperty("class", "danger")
        add_btn.clicked.connect(self.add_movie_record)
        edit_btn.clicked.connect(self.edit_movie_record)
        delete_btn.clicked.connect(self.delete_movie_record)
        refresh_btn.clicked.connect(self.refresh_admin_movies)
        delete_btn.setMaximumWidth(90)
        action_controls.addWidget(add_btn)
        action_controls.addWidget(edit_btn)
        action_controls.addWidget(refresh_btn)
        action_controls.addStretch()
        action_controls.addWidget(QLabel("更改将保存至 SQLite 数据库并在重启后保留；原始 MovieLens 数据文件保持不变。"))
        action_controls.addWidget(delete_btn)
        self.admin_movie_table = QTableWidget()
        self.admin_movie_table.cellClicked.connect(self.update_admin_movie_detail)
        self.movie_result_label = QLabel("结果：全部电影")
        self.movie_result_label.setStyleSheet("font-weight:700; color:#BFDBFE;")
        self.movie_detail_label = QLabel("点击表格中的一行以预览电影详情。")
        self.movie_detail_label.setWordWrap(True)
        self.movie_detail_label.setMinimumHeight(130)
        self.movie_detail_label.setStyleSheet("background:#ffffff; border:1px solid #eeeeee; border-radius:10px; padding:16px; color:#0f0f0f;")
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.admin_movie_table)
        split.addWidget(self.movie_detail_label)
        split.setSizes([720, 420])
        layout.addLayout(controls)
        layout.addLayout(action_controls)
        layout.addWidget(self.movie_result_label)
        layout.addWidget(split, 1)
        self.refresh_admin_movies()

    def selected_admin_movie_id(self) -> int | None:
        row = self.admin_movie_table.currentRow()
        if not hasattr(self, "current_admin_movies") or row < 0 or row >= len(self.current_admin_movies):
            return None
        return int(self.current_admin_movies.iloc[row]["Movie ID"])

    def add_movie_record(self) -> None:
        title, ok = QInputDialog.getText(self, "添加电影", "电影标题：")
        if not ok or not title.strip():
            return
        year, ok = QInputDialog.getInt(self, "添加电影", "上映年份：", 1998, 1900, 2030)
        if not ok:
            return
        genre, ok = QInputDialog.getItem(self, "添加电影", "主要类型：", self.backend.genre_columns(), 0, False)
        if not ok:
            return
        new_id = int(self.backend.movies["movie_id"].max()) + 1
        row = {col: 0 for col in self.backend.movies.columns}
        row.update(
            {
                "movie_id": new_id,
                "title": f"{title.strip()} ({year})",
                "release_year": year,
                "release_date": "",
                "video_release_date": "",
                "imdb_url": "",
            }
        )
        if genre in row:
            row[genre] = 1
        db.add_movie(row, administrator=getattr(self, "admin_username", "admin"))
        self.backend.refresh_movies()
        self.admin_movie_search.setText(str(new_id))
        self.refresh_admin_movies()
        self.movie_result_label.setText(f"电影 #{new_id} 已添加并保存至数据库。共找到 1 条结果")

    def edit_movie_record(self) -> None:
        movie_id = self.selected_admin_movie_id()
        if movie_id is None:
            QMessageBox.information(self, "编辑电影", "请先选择一部电影。")
            return
        current = self.backend.movies.loc[self.backend.movies["movie_id"] == movie_id].iloc[0]
        title, ok = QInputDialog.getText(self, "编辑电影", "电影标题：", text=str(current["title"]))
        if not ok or not title.strip():
            return
        db.update_movie(
            movie_id,
            {"title": title.strip()},
            administrator=getattr(self, "admin_username", "admin"),
        )
        self.backend.refresh_movies()
        self.refresh_admin_movies()
        self.movie_result_label.setText(f"电影 #{movie_id} 已编辑并保存至数据库。")

    def delete_movie_record(self) -> None:
        movie_id = self.selected_admin_movie_id()
        if movie_id is None:
            QMessageBox.information(self, "删除电影", "请先选择一部电影。")
            return
        title = self.backend.catalog.loc[self.backend.catalog["Movie ID"] == movie_id, "Title"].iloc[0]
        confirm = QMessageBox.question(
            self,
            "删除电影",
            f"确定要从电影库中删除《{title}》吗？\n原始 MovieLens 数据文件不会被修改。",
        )
        if confirm != QMessageBox.Yes:
            return
        db.delete_movie(movie_id, administrator=getattr(self, "admin_username", "admin"))
        self.backend.refresh_movies()
        self.refresh_admin_movies()
        self.movie_result_label.setText(f"电影 #{movie_id} 已从数据库中删除。")

    def refresh_admin_movies(self) -> None:
        data = self.backend.search_movies(
            self.admin_movie_search.text(),
            self.admin_movie_genre.currentText(),
            self.admin_movie_sort.currentText(),
        )
        self.current_admin_movies = data
        self.movie_result_label.setText(f"共找到 {len(data):,} 条结果")
        set_table_data(self.admin_movie_table, data, max_rows=500)
        if len(data) == 1:
            self.movie_detail_label.setText(self.backend.movie_detail_text(int(data.iloc[0]["Movie ID"])))
        elif data.empty:
            self.movie_detail_label.setText("没有符合当前筛选条件的电影。")
        else:
            self.movie_detail_label.setText("点击表格中的一行以预览电影详情。")

    def update_admin_movie_detail(self, row: int, _col: int) -> None:
        if hasattr(self, "current_admin_movies") and row < len(self.current_admin_movies):
            movie_id = int(self.current_admin_movies.iloc[row]["Movie ID"])
            self.movie_detail_label.setText(self.backend.movie_detail_text(movie_id))

    def _build_admin_recommendation_tab(self) -> None:
        layout = QVBoxLayout(self.admin_recommendation_tab)
        layout.setSpacing(24)
        title = QLabel("个性化 Top-N 推荐")
        title.setStyleSheet("font-size:20px; font-weight:600;")
        controls = QHBoxLayout()
        self.admin_rec_user = QSpinBox()
        self.admin_rec_user.setRange(
            int(self.backend.ratings["user_id"].min()),
            int(self.backend.ratings["user_id"].max()),
        )
        self.admin_rec_user.setValue(2)
        self.admin_rec_algorithm = QComboBox()
        self.admin_rec_algorithm.addItems(["Item-based CF", "User-based CF", "SVD"])
        self.admin_rec_algorithm.setCurrentText("SVD")
        self.admin_rec_topn = QComboBox()
        self.admin_rec_topn.addItems(["5", "10", "20"])
        self.admin_rec_topn.setCurrentText("10")
        generate_btn = QPushButton(T["btn_generate_recommendations"])
        generate_btn.setProperty("class", "primary")
        generate_btn.clicked.connect(self.refresh_admin_recommendations)
        for label, widget in [
            (T["label_user_id"], self.admin_rec_user),
            (T["label_algorithm"], self.admin_rec_algorithm),
            (T["label_recommendation_count"], self.admin_rec_topn),
        ]:
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(generate_btn)
        self.admin_rec_summary = QLabel("选择一个用户并点击“生成推荐”以显示个性化推荐结果。")
        self.admin_rec_summary.setWordWrap(True)
        self.admin_rec_summary.setStyleSheet(
            "background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; padding:16px; color:#606060;"
        )
        self.admin_rec_table = QTableWidget()
        set_recommendation_table(
            self.admin_rec_table,
            pd.DataFrame(
                [
                    {
                        "Rank": "--",
                        "Movie ID": "--",
                        "Title": "点击“生成推荐”",
                        "Recommendation Score": "--",
                        "Raw Model Score": "--",
                        "Average Rating": "--",
                        "Confidence": "--",
                        "Neighbor Count": "--",
                        "Average Similarity": "--",
                        "Genres": "--",
                        "Reason": "此页面将显示实际的 Top-N 推荐结果。",
                    }
                ]
            ),
            max_rows=1,
        )
        layout.addWidget(title)
        layout.addLayout(controls)
        layout.addWidget(self.admin_rec_summary)
        layout.addWidget(QLabel("Top-N 推荐结果"))
        layout.addWidget(self.admin_rec_table, 1)

    def refresh_admin_recommendations(self) -> None:
        settings = ModelSettings(
            self.admin_rec_algorithm.currentText(),
            self.admin_k_spin.value() if hasattr(self, "admin_k_spin") else 20,
            self.admin_metric_combo.currentText() if hasattr(self, "admin_metric_combo") else "cosine",
            self.admin_factor_spin.value() if hasattr(self, "admin_factor_spin") else 50,
        )
        user_id = int(self.admin_rec_user.value())
        top_n = int(self.admin_rec_topn.currentText())
        self.admin_rec_summary.setText(
            f"正在为用户 {user_id} 使用 {settings.algorithm} 生成 Top-{top_n} 推荐..."
        )
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        recs = self.backend.recommendations(user_id, settings, top_n)
        QApplication.restoreOverrideCursor()
        set_recommendation_table(self.admin_rec_table, recs, max_rows=top_n)
        self.admin_rec_summary.setText(
            f"用户ID：{user_id}    Top-{top_n} 推荐    算法：{settings.algorithm}\n"
            "以上电影均为该用户尚未观看的影片，按校准后的预测偏好、置信度及历史评分证据排序。"
        )

    def _build_user_tab(self) -> None:
        layout = QVBoxLayout(self.user_tab)
        layout.setSpacing(24)
        layout.addWidget(QLabel(T["nav_user_management"]))
        self.admin_user_search = QLineEdit()
        self.admin_user_search.setPlaceholderText("搜索用户ID")
        self.admin_user_search.textChanged.connect(self.refresh_admin_users)
        self.user_card_layout = QGridLayout()
        self.admin_user_table = QTableWidget()
        layout.addWidget(self.admin_user_search)
        layout.addLayout(self.user_card_layout)
        layout.addWidget(QLabel("用户详情表"))
        layout.addWidget(self.admin_user_table)
        self.refresh_admin_users()

    def refresh_admin_users(self) -> None:
        data = self.backend.ratings.groupby("user_id").agg(
            Ratings=("rating", "count"),
            Average_Rating=("rating", "mean"),
            Last_Timestamp=("timestamp", "max"),
        )
        data["Last Activity"] = pd.to_datetime(data["Last_Timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        data = data.drop(columns=["Last_Timestamp"]).reset_index()
        data["Average_Rating"] = data["Average_Rating"].round(2)
        data["Activity Tier"] = np.where(data["Ratings"] >= 100, "活跃", "不活跃")
        profiles = []
        for user_id in data["user_id"]:
            profile = self.backend.user_profile(int(user_id))
            profiles.append(
                {
                    "user_id": user_id,
                    "最喜欢的类型": profile["最喜欢的类型"],
                    "用户画像": profile["用户画像"],
                }
            )
        data = data.merge(pd.DataFrame(profiles), on="user_id", how="left")
        query = self.admin_user_search.text().strip()
        if query:
            data = data[data["user_id"].astype(str).str.contains(query)]
        while self.user_card_layout.count():
            item = self.user_card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for idx, row in enumerate(data.sort_values("Ratings", ascending=False).head(6).itertuples(index=False)):
            card = QLabel(
                f"<b>用户 #{int(row.user_id)}</b><br>"
                f"评分数：{int(row.Ratings):,}<br>"
                f"平均分：{float(row.Average_Rating):.2f}<br>"
                f"最喜欢的类型：{row._5}<br>"
                f"用户画像：{row._6}"
            )
            card.setWordWrap(True)
            card.setStyleSheet("background:#ffffff; border:1px solid #eeeeee; border-radius:10px; padding:16px;")
            self.user_card_layout.addWidget(card, idx // 3, idx % 3)
        set_table_data(self.admin_user_table, data.sort_values("Ratings", ascending=False), max_rows=500)

    def _build_rating_tab(self) -> None:
        layout = QVBoxLayout(self.rating_tab)
        layout.setSpacing(24)
        layout.addWidget(QLabel("评分记录分析"))
        controls = QHBoxLayout()
        self.rating_user_filter = QLineEdit()
        self.rating_movie_filter = QLineEdit()
        self.rating_score_filter = QComboBox()
        self.rating_score_filter.addItems(["All", "1", "2", "3", "4", "5"])
        filter_btn = QPushButton("筛选评分")
        filter_btn.setProperty("class", "secondary")
        filter_btn.clicked.connect(self.refresh_rating_records)
        for label, widget in [
            (T["label_user_id"], self.rating_user_filter),
            (T["label_movie_id"], self.rating_movie_filter),
            ("分数", self.rating_score_filter),
        ]:
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(filter_btn)
        self.rating_table = QTableWidget()
        self.rating_quality = QLabel()
        self.rating_summary_label = QLabel("筛选结果：全部")
        self.rating_summary_label.setStyleSheet("font-weight:700; color:#BFDBFE;")
        self.rating_cards = QHBoxLayout()
        self.rating_chart = ChartWidget()
        self.rating_user_activity_chart = ChartWidget()
        self.rating_movie_pop_chart = ChartWidget()
        self.rating_genre_chart = ChartWidget()
        rating_chart_grid = QGridLayout()
        rating_chart_grid.addWidget(self.rating_chart, 0, 0)
        rating_chart_grid.addWidget(self.rating_user_activity_chart, 0, 1)
        rating_chart_grid.addWidget(self.rating_movie_pop_chart, 1, 0)
        rating_chart_grid.addWidget(self.rating_genre_chart, 1, 1)
        layout.addLayout(controls)
        layout.addWidget(self.rating_quality)
        layout.addWidget(self.rating_summary_label)
        layout.addLayout(self.rating_cards)
        layout.addLayout(rating_chart_grid)
        layout.addWidget(QLabel("评分详情表"))
        layout.addWidget(self.rating_table)
        self.refresh_rating_records()

    def refresh_rating_records(self) -> None:
        data = self.backend.ratings.copy()
        if self.rating_user_filter.text().strip():
            user_key = self.rating_user_filter.text().strip()
            if user_key.isdigit():
                data = data[data["user_id"] == int(user_key)]
            else:
                data = data[data["user_id"].astype(str).str.contains(user_key)]
        if self.rating_movie_filter.text().strip():
            movie_key = self.rating_movie_filter.text().strip()
            if movie_key.isdigit():
                data = data[data["movie_id"] == int(movie_key)]
            else:
                data = data[data["movie_id"].astype(str).str.contains(movie_key)]
        if self.rating_score_filter.currentText() != "All":
            data = data[data["rating"] == int(self.rating_score_filter.currentText())]
        data = data.copy()
        if "timestamp" in data.columns:
            data["Date"] = pd.to_datetime(data["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
            data = data[["user_id", "movie_id", "rating", "Date"]]
        missing = int(self.backend.ratings.isna().sum().sum())
        abnormal = int((~self.backend.ratings["rating"].between(1, 5)).sum())
        self.rating_quality.setText(f"缺失值：{missing}    异常评分：{abnormal}")
        set_table_data(self.rating_table, data, max_rows=500)
        while self.rating_cards.count():
            item = self.rating_cards.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not data.empty:
            summary_values = {
                "筛选行数": f"{len(data):,}",
                "平均评分": f"{data['rating'].mean():.2f}",
                "用户数": f"{data['user_id'].nunique():,}",
                "电影数": f"{data['movie_id'].nunique():,}",
            }
            for title, value in summary_values.items():
                self.rating_cards.addWidget(metric_label(title, value))
            self.rating_chart.bar(data["rating"].value_counts().sort_index(), "筛选后的评分分布", "评分", "数量")
            user_counts = data.groupby("user_id").size()
            bins = pd.cut(user_counts, bins=min(10, max(1, user_counts.nunique())))
            self.rating_user_activity_chart.bar(
                bins.value_counts().sort_index(),
                "用户评分次数分布",
                "每用户评分数",
                "用户数",
            )
            movie_counts = data.groupby("movie_id").size().sort_values(ascending=False).head(10)
            movie_titles = self.backend.catalog.set_index("Movie ID")["Title"]
            movie_counts.index = [str(movie_titles.get(int(movie_id), movie_id))[:22] for movie_id in movie_counts.index]
            self.rating_movie_pop_chart.bar(movie_counts, "筛选结果中的热门电影", "电影", "评分数")
            genre_counts = {}
            for movie_id in data["movie_id"].drop_duplicates():
                for genre in self.backend.genre_text(int(movie_id)).split(", "):
                    if genre and genre != "未知":
                        genre_counts[genre] = genre_counts.get(genre, 0) + int((data["movie_id"] == movie_id).sum())
            self.rating_genre_chart.bar(
                pd.Series(genre_counts).sort_values(ascending=False).head(10),
                "评分中的类型分布",
                "类型",
                "评分数",
            )
            self.rating_summary_label.setText(
                f"筛选行数：{len(data):,}    平均评分：{data['rating'].mean():.2f}    "
                f"用户数：{data['user_id'].nunique():,}    电影数：{data['movie_id'].nunique():,}"
            )
        else:
            for title in ["筛选行数", "平均评分", "用户数", "电影数"]:
                self.rating_cards.addWidget(metric_label(title, "0"))
            self.rating_summary_label.setText("筛选行数：0")
            empty = pd.Series({"无数据": 0})
            self.rating_chart.bar(empty, "筛选后的评分分布")
            self.rating_user_activity_chart.bar(empty, "用户评分次数分布")
            self.rating_movie_pop_chart.bar(empty, "筛选结果中的热门电影")
            self.rating_genre_chart.bar(empty, "评分中的类型分布")

    def _build_algorithm_tab(self) -> None:
        outer = QVBoxLayout(self.algorithm_tab)
        outer.setSpacing(24)
        outer.addWidget(QLabel(T["nav_algorithm_config"]))
        model_cards = QGridLayout()
        descriptions = [
            ("User-based CF", "通过相似用户预测偏好。", "优点：直观、可解释性强。局限：用户量大时速度较慢。"),
            ("Item-based CF", "基于评分模式查找相似电影。", "优点：稳定、计算效率高。局限：对新电影的推荐效果较弱。"),
            ("SVD", "学习用户与电影的潜在因子。", "优点：预测性能最佳。局限：可解释性较弱。"),
        ]
        for idx, (name, desc, note) in enumerate(descriptions):
            card = QLabel(f"<b>{name}</b><br>{desc}<br><span style='color:#606060'>{note}</span>")
            card.setWordWrap(True)
            card.setStyleSheet("background:#ffffff; border:1px solid #eeeeee; border-radius:10px; padding:16px;")
            model_cards.addWidget(card, 0, idx)
        outer.addLayout(model_cards)
        layout = QFormLayout()
        self.admin_algorithm_combo = QComboBox()
        self.admin_algorithm_combo.addItems(["Item-based CF", "User-based CF", "SVD"])
        self.admin_algorithm_combo.currentTextChanged.connect(self.update_admin_parameter_visibility)
        self.admin_k_spin = QSpinBox()
        self.admin_k_spin.setRange(5, 50)
        self.admin_k_spin.setValue(20)
        self.admin_factor_spin = QSpinBox()
        self.admin_factor_spin.setRange(10, 80)
        self.admin_factor_spin.setValue(50)
        self.admin_metric_combo = QComboBox()
        self.admin_metric_combo.addItems(["cosine", "pearson"])
        save_btn = QPushButton("保存当前算法配置")
        save_btn.setProperty("class", "admin")
        save_btn.clicked.connect(self.save_algorithm_settings)
        self.admin_algorithm_label = QLabel("当前算法")
        self.admin_k_label = QLabel(T["label_neighbors_k"])
        self.admin_factor_label = QLabel(T["label_svd_factors"])
        self.admin_metric_label = QLabel(T["label_similarity_method"])
        layout.addRow(self.admin_algorithm_label, self.admin_algorithm_combo)
        layout.addRow(self.admin_k_label, self.admin_k_spin)
        layout.addRow(self.admin_factor_label, self.admin_factor_spin)
        layout.addRow(self.admin_metric_label, self.admin_metric_combo)
        layout.addWidget(save_btn)
        self.algorithm_info = QTextEdit()
        self.algorithm_info.setReadOnly(True)
        self.algorithm_info.setMinimumHeight(180)
        self.algorithm_info.setText(
            "参数说明\n"
            "K：UserCF 和 ItemCF 使用的最近邻数量。\n"
            "相似度计算方法：基于评分向量的余弦相似度或皮尔逊相关系数。\n"
            "SVD 因子数：SVD 嵌入向量的维度。\n\n"
            "推荐流程\n"
            "输入用户 -> 选择模型 -> 对未观看电影打分 -> 置信度调整 -> Top-N 排序\n\n"
            "冷启动处理\n"
            "若用户或电影历史数据不足，系统将回退至全局、用户、物品或热度统计的基线方法。"
        )
        self.algorithm_perf = QLabel("当前性能摘要：运行评估以查看 MAE/RMSE 和 Top-N 指标。")
        self.algorithm_perf.setWordWrap(True)
        outer.addLayout(layout)
        outer.addWidget(self.algorithm_info)
        outer.addWidget(self.algorithm_perf)
        self.update_admin_parameter_visibility()

    def save_algorithm_settings(self) -> None:
        self.current_admin_settings = ModelSettings(
            self.admin_algorithm_combo.currentText(),
            self.admin_k_spin.value(),
            self.admin_metric_combo.currentText(),
            self.admin_factor_spin.value(),
        )
        self.algorithm_perf.setText("正在使用当前配置重建模型...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.backend.model(self.current_admin_settings)
        QApplication.restoreOverrideCursor()
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg = (
            "模型重建成功\n"
            f"当前配置：算法={self.current_admin_settings.algorithm}, "
            f"K={self.current_admin_settings.k}, 相似度={self.current_admin_settings.metric}, "
            f"因子数={self.current_admin_settings.factors}\n"
            f"最近更新：{updated_at}"
        )
        self.algorithm_perf.setText(msg)
        self.admin_model_status.setText(
            f"<span style='font-size:32px; font-weight:700; color:#0f0f0f'>{self.current_admin_settings.algorithm}</span>"
            "<br><span style='font-size:12px; color:#606060'>当前模型</span>"
        )
        self.admin_time_status.setText(
            f"<span style='font-size:32px; font-weight:700; color:#0f0f0f'>{updated_at.split()[1]}</span>"
            "<br><span style='font-size:12px; color:#606060'>最近一次模型重建</span>"
        )

    def update_admin_parameter_visibility(self) -> None:
        algorithm = self.admin_algorithm_combo.currentText()
        uses_neighbors = algorithm in {"Item-based CF", "User-based CF"}
        uses_factors = algorithm == "SVD"
        self.admin_k_label.setVisible(uses_neighbors)
        self.admin_k_spin.setVisible(uses_neighbors)
        self.admin_metric_label.setVisible(uses_neighbors)
        self.admin_metric_combo.setVisible(uses_neighbors)
        self.admin_factor_label.setVisible(uses_factors)
        self.admin_factor_spin.setVisible(uses_factors)

    def _build_eval_tab(self) -> None:
        layout = QVBoxLayout(self.eval_tab)
        layout.setSpacing(24)
        layout.addWidget(QLabel(T["nav_model_evaluation"]))
        button_row = QHBoxLayout()
        run_btn = QPushButton("运行 MAE / RMSE 评估")
        run_btn.setProperty("class", "primary")
        run_btn.clicked.connect(self.refresh_evaluation)
        topn_btn = QPushButton("运行 Top-N 评估")
        topn_btn.setProperty("class", "primary")
        topn_btn.clicked.connect(self.refresh_topn_evaluation)
        sig_btn = QPushButton("运行显著性检验")
        sig_btn.setProperty("class", "primary")
        sig_btn.clicked.connect(self.refresh_significance)
        svd_btn = QPushButton("运行 SVD 因子数实验")
        svd_btn.setProperty("class", "secondary")
        svd_btn.clicked.connect(self.refresh_svd_experiment)
        button_row.addWidget(run_btn)
        button_row.addWidget(topn_btn)
        button_row.addWidget(sig_btn)
        button_row.addWidget(svd_btn)
        self.eval_table = QTableWidget()
        self.topn_table = QTableWidget()
        self.significance_table = QTableWidget()
        self.svd_experiment_table = QTableWidget()
        self.eval_kpi_layout = QHBoxLayout()
        for title in ["最优 RMSE", "最优 NDCG@10", "最优 Recall@10", "p-value"]:
            self.eval_kpi_layout.addWidget(metric_label(title, "--"))
        set_table_data(self.eval_table, pd.DataFrame([{"Status": "点击“运行 MAE / RMSE 评估”以生成结果。"}]))
        set_table_data(self.topn_table, pd.DataFrame([{"Status": "点击“运行 Top-N 评估”以生成结果。"}]))
        set_table_data(self.significance_table, pd.DataFrame([{"Status": "点击“运行显著性检验”以生成结果。"}]))
        set_table_data(self.svd_experiment_table, pd.DataFrame([{"Status": "点击“运行 SVD 因子数实验”以生成结果。"}]))
        self.eval_chart = ChartWidget()
        self.topn_chart = ChartWidget()
        self.svd_experiment_chart = ChartWidget()
        layout.addLayout(button_row)
        layout.addLayout(self.eval_kpi_layout)
        layout.addWidget(QLabel("评分预测指标"))
        layout.addWidget(self.eval_table)
        layout.addWidget(self.eval_chart)
        layout.addWidget(QLabel("Top-N 排序指标"))
        layout.addWidget(self.topn_table)
        layout.addWidget(self.topn_chart)
        layout.addWidget(QLabel("SVD 参数实验"))
        layout.addWidget(self.svd_experiment_table)
        layout.addWidget(self.svd_experiment_chart)
        layout.addWidget(QLabel("统计显著性检验"))
        layout.addWidget(self.significance_table)

    def refresh_evaluation(self) -> None:
        self.eval_table.setToolTip("正在运行 MAE / RMSE 评估...")
        set_table_data(self.eval_table, pd.DataFrame([{"Status": "正在运行 MAE / RMSE 评估..."}]))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        data = self.backend.evaluate_models()
        QApplication.restoreOverrideCursor()
        set_table_data(self.eval_table, data, max_rows=10)
        self.eval_table.setToolTip("已完成。")
        self.eval_chart.bar(data.set_index("Algorithm")["RMSE"], "各算法 RMSE 对比")
        best = data.sort_values("RMSE").iloc[0]
        self.set_eval_kpi(0, "最优 RMSE", f"{best['RMSE']}")
        self.algorithm_perf.setText(
            f"当前性能摘要：最优 RMSE = {best['Algorithm']} "
            f"({best['RMSE']})；该表基于按时间切分的留出评分样本计算 MAE/RMSE。"
        )

    def refresh_topn_evaluation(self) -> None:
        self.topn_table.setToolTip("正在运行 Top-N 评估...")
        set_table_data(self.topn_table, pd.DataFrame([{"Status": "正在运行 Top-N 评估..."}]))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        data = self.backend.topn_evaluation()
        QApplication.restoreOverrideCursor()
        set_table_data(self.topn_table, data, max_rows=10)
        self.topn_table.setToolTip("已完成。")
        self.topn_chart.bar(data.set_index("Algorithm")["NDCG@10"], "各算法 NDCG@10 对比")
        best_ndcg = data.sort_values("NDCG@10", ascending=False).iloc[0]
        best_recall = data.sort_values("Recall@10", ascending=False).iloc[0]
        self.set_eval_kpi(1, "最优 NDCG@10", f"{best_ndcg['NDCG@10']}")
        self.set_eval_kpi(2, "最优 Recall@10", f"{best_recall['Recall@10']}")

    def refresh_significance(self) -> None:
        self.significance_table.setToolTip("正在运行显著性检验...")
        set_table_data(self.significance_table, pd.DataFrame([{"Status": "正在运行显著性检验..."}]))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        data = self.backend.significance_summary()
        QApplication.restoreOverrideCursor()
        set_table_data(self.significance_table, data, max_rows=5)
        self.significance_table.setToolTip("已完成。")
        self.set_eval_kpi(3, "p-value", str(data.iloc[0]["p-value"]))

    def refresh_svd_experiment(self) -> None:
        self.svd_experiment_table.setToolTip("正在运行 SVD 因子数实验...")
        set_table_data(self.svd_experiment_table, pd.DataFrame([{"Status": "正在运行 SVD 因子数实验..."}]))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        data = self.backend.svd_factor_experiment()
        QApplication.restoreOverrideCursor()
        set_table_data(self.svd_experiment_table, data, max_rows=10)
        self.svd_experiment_table.setToolTip("已完成。")
        self.svd_experiment_chart.line(data.set_index("Factors")["RMSE"], "SVD 因子数与 RMSE 关系", "因子数", "RMSE")

    def set_eval_kpi(self, index: int, title: str, value: str) -> None:
        item = self.eval_kpi_layout.takeAt(index)
        if item and item.widget():
            item.widget().deleteLater()
        self.eval_kpi_layout.insertWidget(index, metric_label(title, value))

    def _build_stats_tab(self) -> None:
        layout = QVBoxLayout(self.stats_tab)
        self.most_rated_table = QTableWidget()
        self.highest_rated_table = QTableWidget()
        self.genre_pop_chart = ChartWidget()
        self.user_activity_chart = ChartWidget()
        self.coverage_label = QLabel("Top-N 覆盖率与多样性：尚未计算。")
        coverage_btn = QPushButton("计算 Top-N 覆盖率 / 多样性")
        coverage_btn.setProperty("class", "secondary")
        coverage_btn.clicked.connect(self.refresh_coverage_diversity)
        rec_stats_btn = QPushButton("计算推荐系统统计指标")
        rec_stats_btn.setProperty("class", "primary")
        rec_stats_btn.clicked.connect(self.refresh_recommendation_statistics)
        self.recommendation_stats_table = QTableWidget()
        set_table_data(
            self.recommendation_stats_table,
            pd.DataFrame([{"Status": "点击“计算推荐系统统计指标”以生成 RMSE、MAE、最优模型、覆盖率和多样性数据。"}]),
        )
        layout.addWidget(QLabel("评分最多的电影"))
        layout.addWidget(self.most_rated_table)
        layout.addWidget(QLabel("评分最高的电影"))
        layout.addWidget(self.highest_rated_table)
        chart_row = QHBoxLayout()
        chart_row.addWidget(self.genre_pop_chart, 2)
        chart_row.addWidget(self.user_activity_chart, 2)
        layout.addLayout(chart_row)
        quality_row = QHBoxLayout()
        quality_row.addWidget(coverage_btn)
        quality_row.addWidget(rec_stats_btn)
        quality_row.addWidget(self.coverage_label, 1)
        layout.addLayout(quality_row)
        layout.addWidget(QLabel("推荐系统统计"))
        layout.addWidget(self.recommendation_stats_table)
        set_table_data(self.most_rated_table, self.backend.popular_movies("Number of Ratings", 20), max_rows=20)
        set_table_data(self.highest_rated_table, self.backend.popular_movies("Average Rating", 20), max_rows=20)
        genre_counts = pd.Series(
            {genre: int(self.backend.movies[genre].sum()) for genre in self.backend.genre_columns()}
        ).sort_values(ascending=False)
        self.genre_pop_chart.bar(genre_counts, "电影类型热度")
        user_activity = self.backend.ratings.groupby("user_id").size().sort_values(ascending=False).head(20)
        self.user_activity_chart.bar(user_activity, "最活跃用户", "用户ID", "评分数")

    def refresh_coverage_diversity(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        stats = self.backend.coverage_and_diversity()
        QApplication.restoreOverrideCursor()
        self.coverage_label.setText(
            "    ".join(f"{key}: {value}" for key, value in stats.items())
            + "    计算方法：基于抽样用户的 SVD Top-10 推荐列表。"
        )

    def refresh_recommendation_statistics(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        data = self.backend.recommendation_statistics()
        QApplication.restoreOverrideCursor()
        set_table_data(self.recommendation_stats_table, data, max_rows=10)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FilmTrace · MovieLens 推荐系统")
        self.resize(1360, 860)
        self.backend = MovieLensBackend()
        self.stack = QStackedWidget()
        self.pages = {
            "welcome": WelcomePage(self),
            "user": UserPage(self.backend),
            "admin": AdminPage(self.backend),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._build_top_nav())
        shell_layout.addWidget(self.stack, 1)
        self.setCentralWidget(shell)
        self.show_page("welcome")
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#f9f9f9; color:#0f0f0f; font-family:Roboto, "Segoe UI", Arial; font-size:14px; }
            #topNav { background:#ffffff; border-bottom:1px solid #c6c6c6; }
            #brandLabel { font-size:20px; font-weight:600; color:#0f0f0f; padding-left:16px; }
            #topSearch { background:#ffffff; color:#0f0f0f; border:1px solid #c6c6c6; border-radius:22px; padding:9px 16px; min-width:620px; min-height:24px; }
            QGroupBox { background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; margin-top:14px; padding:16px; }
            QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 6px; color:#0f0f0f; font-weight:600; font-size:16px; }
            QLabel { color:#0f0f0f; }
            QPushButton { background:rgba(0,0,0,0.05); color:#0f0f0f; border:0; border-radius:18px; padding:8px 16px; font-weight:500; }
            QPushButton:hover { background:rgba(0,0,0,0.08); }
            QPushButton[class="secondary"] { background:rgba(0,0,0,0.05); color:#0f0f0f; }
            QPushButton[class="admin"] { background:rgba(0,0,0,0.05); color:#0f0f0f; }
            QPushButton[class="danger"] { background:rgba(0,0,0,0.05); color:#0f0f0f; }
            QPushButton[class="primary"] { background:#2563EB; color:#ffffff; }
            QPushButton:disabled { background:#eeeeee; color:#909090; }
            QLineEdit, QSpinBox, QComboBox, QTextEdit { background:#ffffff; color:#0f0f0f; border:1px solid #c6c6c6; border-radius:0px; padding:7px; }
            QTableWidget { background:#ffffff; color:#0f0f0f; gridline-color:#eeeeee; alternate-background-color:#fafafa; selection-background-color:rgba(0,0,0,0.05); font-size:14px; border:1px solid #e5e5e5; }
            QHeaderView::section { background:#ffffff; color:#606060; padding:8px; border:0px; border-bottom:1px solid #c6c6c6; font-weight:500; }
            QTabWidget::pane { border:0px; }
            QTabBar::tab { background:#ffffff; color:#0f0f0f; padding:12px 18px; min-width:184px; border-radius:10px; text-align:left; }
            QTabBar::tab:selected { background:rgba(0,0,0,0.05); color:#0f0f0f; }
            QListWidget#sideNav { background:#f9f9f9; border:0px; padding:8px; font-size:14px; }
            QListWidget#sideNav::item { color:#0f0f0f; padding:12px 14px; border-radius:10px; min-height:22px; }
            QListWidget#sideNav::item:selected { background:rgba(0,0,0,0.05); font-weight:500; }
            QListWidget#sideNav::item:hover { background:rgba(0,0,0,0.05); }
            QScrollArea { background:#f8fafc; border:0px; }
            """
        )

    def _build_top_nav(self) -> QWidget:
        nav = QWidget()
        nav.setObjectName("topNav")
        nav.setFixedHeight(56)
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(12, 0, 16, 0)
        layout.setSpacing(16)
        brand = QLabel("☰  FilmTrace")
        brand.setObjectName("brandLabel")
        search = QLineEdit()
        search.setObjectName("topSearch")
        search.setPlaceholderText("搜索电影...")
        search.returnPressed.connect(lambda: self.show_page("user"))
        user_btn = QPushButton("浏览电影")
        admin_btn = QPushButton(T["nav_admin"])
        user_btn.clicked.connect(lambda: self.show_page("user"))
        admin_btn.clicked.connect(lambda: self.show_page("admin"))
        layout.addWidget(brand)
        layout.addStretch(1)
        layout.addWidget(search, 2)
        layout.addStretch(1)
        layout.addWidget(user_btn)
        layout.addWidget(admin_btn)
        return nav

    def show_page(self, name: str) -> None:
        self.stack.setCurrentWidget(self.pages[name])


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
