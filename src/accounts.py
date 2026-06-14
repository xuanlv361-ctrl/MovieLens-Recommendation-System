"""Account system for the FilmTrace user platform.

This module is independent of the legacy MovieLens `users`/`ratings` tables
in `src.db` (which are read-only reference copies of the dataset). Real
visitors register an `accounts` row here, optionally linked to a MovieLens
`user_id` (`movielens_user_id`) so the existing recommendation models can be
reused for that account's "为你推荐" results.

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes (stdlib `hashlib`,
no extra dependency).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .db import DB_PATH, get_connection, log_admin_action

_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: str | None = None) -> str:
    """Return `salt$hash` for storage; generates a random salt if not given."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Return True if the plaintext password matches the stored hash."""
    salt, _, _ = stored_hash.partition("$")
    if not salt:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored_hash)


class AccountError(ValueError):
    """Raised for user-facing registration/login validation errors."""


def register_account(
    username: str,
    email: str,
    password: str,
    movielens_user_id: int | None = None,
    db_path: Path | str = DB_PATH,
) -> int:
    """Create a new account. Raises `AccountError` with a Chinese message on failure."""
    username = username.strip()
    email = email.strip().lower()
    if not username or not email or not password:
        raise AccountError("用户名、邮箱和密码不能为空。")

    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT 1 FROM accounts WHERE username = ? OR email = ?", (username, email)
        ).fetchone()
        if existing is not None:
            raise AccountError("该用户名或邮箱已被注册。")

        cur = conn.execute(
            "INSERT INTO accounts (username, email, password_hash, movielens_user_id, "
            "is_active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (
                username,
                email,
                hash_password(password),
                movielens_user_id,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def authenticate_account(username_or_email: str, password: str, db_path: Path | str = DB_PATH) -> dict | None:
    """Return the account row (dict) if credentials are valid and active, else None."""
    identifier = username_or_email.strip()
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM accounts WHERE username = ? OR email = ?",
            (identifier, identifier.lower()),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    if not row["is_active"]:
        return None
    return dict(row)


def get_account(account_id: int, db_path: Path | str = DB_PATH) -> dict | None:
    """Return the account row for the given username, or None if it does not exist."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM accounts WHERE account_id = ?", (int(account_id),)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def update_password(account_id: int, new_password: str, db_path: Path | str = DB_PATH) -> None:
    """Update the stored password hash for the given account."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE accounts SET password_hash = ? WHERE account_id = ?",
            (hash_password(new_password), int(account_id)),
        )
        conn.commit()
    finally:
        conn.close()


def set_account_active(account_id: int, is_active: bool, administrator: str, db_path: Path | str = DB_PATH) -> None:
    """Enable/disable an account and record the action in `admin_audit_log`."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE accounts SET is_active = ? WHERE account_id = ?",
            (1 if is_active else 0, int(account_id)),
        )
        operation = "enable_user" if is_active else "disable_user"
        log_admin_action(operation, administrator, f"account_id={account_id}", conn=conn)
        conn.commit()
    finally:
        conn.close()


def list_accounts(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Return all accounts with rating/favorite counts for the admin user-management page."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT a.account_id, a.username, a.email, a.created_at, a.is_active,
                   a.movielens_user_id,
                   COALESCE(u.n_ratings, 0) AS n_ratings,
                   (SELECT COUNT(*) FROM favorites f WHERE f.account_id = a.account_id) AS n_favorites
            FROM accounts a
            LEFT JOIN users u ON u.user_id = a.movielens_user_id
            ORDER BY a.account_id
            """
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(row) for row in rows])


# --- Favorites -------------------------------------------------------------


def add_favorite(account_id: int, movie_id: int, db_path: Path | str = DB_PATH) -> None:
    """Add a movie to the account's favorites."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO favorites (account_id, movie_id, created_at) VALUES (?, ?, ?)",
            (int(account_id), int(movie_id), datetime.now(UTC).isoformat(timespec="seconds")),
        )
        conn.commit()
    finally:
        conn.close()


def remove_favorite(account_id: int, movie_id: int, db_path: Path | str = DB_PATH) -> None:
    """Remove a movie from the account's favorites."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "DELETE FROM favorites WHERE account_id = ? AND movie_id = ?",
            (int(account_id), int(movie_id)),
        )
        conn.commit()
    finally:
        conn.close()


def list_favorites(account_id: int, db_path: Path | str = DB_PATH) -> list[int]:
    """Return the list of movie ids favorited by the account."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT movie_id FROM favorites WHERE account_id = ? ORDER BY created_at DESC",
            (int(account_id),),
        ).fetchall()
    finally:
        conn.close()
    return [int(row["movie_id"]) for row in rows]


def is_favorite(account_id: int, movie_id: int, db_path: Path | str = DB_PATH) -> bool:
    """Return True if the movie is in the account's favorites."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM favorites WHERE account_id = ? AND movie_id = ?",
            (int(account_id), int(movie_id)),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


# --- Preferences & profile ---------------------------------------------------


def save_preferences(
    account_id: int, genres: list[str], seed_movie_ids: list[int], db_path: Path | str = DB_PATH
) -> None:
    """Persist the account's selected genre preferences and seed movies."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO user_preferences (account_id, genres, seed_movie_ids, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET genres = excluded.genres, "
            "seed_movie_ids = excluded.seed_movie_ids, updated_at = excluded.updated_at",
            (
                int(account_id),
                json.dumps(genres, ensure_ascii=False),
                json.dumps([int(m) for m in seed_movie_ids]),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_preferences(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Return all stored genre preferences as a DataFrame with `account_id`, `genres` (list)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT account_id, genres FROM user_preferences").fetchall()
    finally:
        conn.close()
    return pd.DataFrame(
        [{"account_id": int(row["account_id"]), "genres": json.loads(row["genres"])} for row in rows],
        columns=["account_id", "genres"],
    )


def get_preferences(account_id: int, db_path: Path | str = DB_PATH) -> dict | None:
    """Return the stored genre preferences for the account, or None if unset."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT genres, seed_movie_ids, updated_at FROM user_preferences WHERE account_id = ?",
            (int(account_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "genres": json.loads(row["genres"]),
        "seed_movie_ids": json.loads(row["seed_movie_ids"]),
        "updated_at": row["updated_at"],
    }


def save_profile(
    account_id: int,
    top_genres: list[str],
    year_range: str,
    preferred_min_rating: float,
    db_path: Path | str = DB_PATH,
) -> None:
    """Persist the computed user profile (top genres, year range, preferred rating)."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO user_profiles (account_id, top_genres, year_range, preferred_min_rating, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET top_genres = excluded.top_genres, "
            "year_range = excluded.year_range, preferred_min_rating = excluded.preferred_min_rating, "
            "updated_at = excluded.updated_at",
            (
                int(account_id),
                json.dumps(top_genres, ensure_ascii=False),
                year_range,
                float(preferred_min_rating),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_profile(account_id: int, db_path: Path | str = DB_PATH) -> dict | None:
    """Return the stored user profile for the account, or None if unset."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT top_genres, year_range, preferred_min_rating, updated_at FROM user_profiles "
            "WHERE account_id = ?",
            (int(account_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "top_genres": json.loads(row["top_genres"]),
        "year_range": row["year_range"],
        "preferred_min_rating": row["preferred_min_rating"],
        "updated_at": row["updated_at"],
    }


def ensure_schema(db_path: Path | str = DB_PATH) -> None:
    """Create the account-related tables if `init_db()` has not run yet."""
    from .db import _SCHEMA, migrate_schema

    conn: sqlite3.Connection = get_connection(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        migrate_schema(conn)
    finally:
        conn.close()
