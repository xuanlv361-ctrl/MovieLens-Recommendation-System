"""SQLite persistence layer for the MovieLens demo apps.

Provides durable storage for the movie catalog (so admin Add/Edit/Delete
operations survive restarts), plus reference copies of the `users` and
`ratings` tables and an `admin_audit_log` that records every administrator
mutation (timestamp, operation, administrator, target object).

The heavy analytics/modeling pipelines continue to read ratings via
`src.data_loader` (bulk CSV) for performance; this module is the system of
record for admin-editable data and audit history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED

DB_PATH = DATA_PROCESSED / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    n_ratings INTEGER NOT NULL,
    avg_rating REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS admin_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    operation TEXT NOT NULL,
    administrator TEXT NOT NULL,
    target TEXT NOT NULL
);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(
    movies_df: pd.DataFrame | None = None,
    ratings_df: pd.DataFrame | None = None,
    db_path: Path | str = DB_PATH,
) -> sqlite3.Connection:
    """Create tables if missing and seed them from the raw dataset on first run."""
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA)

    movie_count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    if movie_count == 0 and movies_df is not None:
        seed_movies(movies_df, conn=conn)

    rating_count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    if rating_count == 0 and ratings_df is not None:
        seed_ratings(ratings_df, conn=conn)

    conn.commit()
    return conn


def seed_movies(movies_df: pd.DataFrame, conn: sqlite3.Connection | None = None) -> None:
    owns_conn = conn is None
    conn = conn or get_connection()
    rows = [
        (int(row["movie_id"]), str(row["title"]), json.dumps(_row_to_jsonable(row)))
        for _, row in movies_df.iterrows()
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO movies (movie_id, title, data) VALUES (?, ?, ?)",
        rows,
    )
    if owns_conn:
        conn.commit()
        conn.close()


def seed_ratings(ratings_df: pd.DataFrame, conn: sqlite3.Connection | None = None) -> None:
    owns_conn = conn is None
    conn = conn or get_connection()
    rating_rows = [
        (int(r.user_id), int(r.movie_id), int(r.rating), int(r.timestamp))
        for r in ratings_df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
        rating_rows,
    )
    user_stats = ratings_df.groupby("user_id")["rating"].agg(["count", "mean"])
    user_rows = [
        (int(user_id), int(stats["count"]), float(stats["mean"]))
        for user_id, stats in user_stats.iterrows()
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO users (user_id, n_ratings, avg_rating) VALUES (?, ?, ?)",
        user_rows,
    )
    if owns_conn:
        conn.commit()
        conn.close()


def _row_to_jsonable(row: pd.Series) -> dict:
    out = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def get_movies_df(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Return the persisted movie catalog as a DataFrame with the original columns."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT data FROM movies ORDER BY movie_id").fetchall()
    finally:
        conn.close()
    records = [json.loads(row["data"]) for row in rows]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def add_movie(row: dict, administrator: str, db_path: Path | str = DB_PATH) -> int:
    """Insert a new movie record and write an audit-log entry. Returns the movie_id."""
    conn = get_connection(db_path)
    try:
        movie_id = int(row["movie_id"])
        conn.execute(
            "INSERT INTO movies (movie_id, title, data) VALUES (?, ?, ?)",
            (movie_id, str(row.get("title", "")), json.dumps(_jsonable_dict(row))),
        )
        log_admin_action("add_movie", administrator, f"movie_id={movie_id}", conn=conn)
        conn.commit()
        return movie_id
    finally:
        conn.close()


def update_movie(
    movie_id: int, updates: dict, administrator: str, db_path: Path | str = DB_PATH
) -> bool:
    """Merge `updates` into the stored movie record and write an audit-log entry."""
    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT data FROM movies WHERE movie_id = ?", (int(movie_id),)
        ).fetchone()
        if existing is None:
            return False
        data = json.loads(existing["data"])
        data.update(_jsonable_dict(updates))
        title = str(data.get("title", ""))
        conn.execute(
            "UPDATE movies SET title = ?, data = ? WHERE movie_id = ?",
            (title, json.dumps(data), int(movie_id)),
        )
        log_admin_action("edit_movie", administrator, f"movie_id={movie_id}", conn=conn)
        conn.commit()
        return True
    finally:
        conn.close()


def delete_movie(movie_id: int, administrator: str, db_path: Path | str = DB_PATH) -> bool:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("DELETE FROM movies WHERE movie_id = ?", (int(movie_id),))
        if cur.rowcount:
            log_admin_action("delete_movie", administrator, f"movie_id={movie_id}", conn=conn)
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def _jsonable_dict(values: dict) -> dict:
    out = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def log_admin_action(
    operation: str,
    administrator: str,
    target: str,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = DB_PATH,
) -> None:
    """Record an admin action in `admin_audit_log` (timestamp, operation, administrator, target)."""
    owns_conn = conn is None
    conn = conn or get_connection(db_path)
    conn.execute(
        "INSERT INTO admin_audit_log (ts, operation, administrator, target) VALUES (?, ?, ?, ?)",
        (datetime.now(UTC).isoformat(timespec="seconds"), operation, administrator, target),
    )
    if owns_conn:
        conn.commit()
        conn.close()


def get_audit_log(limit: int = 200, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT ts, operation, administrator, target FROM admin_audit_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["ts", "operation", "administrator", "target"],
    ).rename(
        columns={
            "ts": "Timestamp",
            "operation": "Operation",
            "administrator": "Administrator",
            "target": "Target",
        }
    )
