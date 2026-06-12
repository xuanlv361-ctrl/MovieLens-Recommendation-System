"""Poster image helpers shared by the Streamlit and PyQt5 demo apps.

MovieLens 100K ships only metadata (titles, genres, IMDb links) and contains
no poster artwork. When a `TMDB_API_KEY` environment variable is configured,
posters are looked up on The Movie Database; otherwise callers fall back to
a deterministic gradient placeholder keyed by the recommendation rank.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"

_TITLE_YEAR_RE = re.compile(r"^(.*)\s\((\d{4})\)\s*$")

# Local poster directory: image filenames are Chinese (or English) movie
# titles, e.g. "玩具总动员.jpg". Using local posters first means the demo
# never depends on TMDb during a defense/presentation.
LOCAL_POSTER_DIR = Path(__file__).resolve().parent.parent / "电影照片"
LOCAL_POSTER_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

POSTER_PALETTES = [
    "linear-gradient(135deg, #2b1d40, #0f1216)",
    "linear-gradient(135deg, #16304b, #101418)",
    "linear-gradient(135deg, #541d1d, #111418)",
    "linear-gradient(135deg, #123f2b, #101418)",
    "linear-gradient(135deg, #3e3217, #101418)",
    "linear-gradient(135deg, #35485a, #101418)",
    "linear-gradient(135deg, #2d3440, #14181c)",
    "linear-gradient(135deg, #49243a, #101418)",
    "linear-gradient(135deg, #1d3a45, #101418)",
    "linear-gradient(135deg, #3b2430, #101418)",
]


def poster_background(rank: int) -> str:
    """Return a deterministic gradient placeholder for the given rank."""
    return POSTER_PALETTES[(rank - 1) % len(POSTER_PALETTES)]


def clean_title_year(title: str) -> tuple[str, int | None]:
    """Split a MovieLens title like 'Toy Story (1995)' into title and year."""
    match = _TITLE_YEAR_RE.match(title.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


@functools.lru_cache(maxsize=1)
def _local_poster_index() -> dict[str, Path]:
    """Map local poster filename stem (without extension) -> file path."""
    if not LOCAL_POSTER_DIR.is_dir():
        return {}
    index: dict[str, Path] = {}
    for path in LOCAL_POSTER_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in LOCAL_POSTER_EXTENSIONS:
            index[path.stem] = path
    return index


def list_local_posters() -> dict[str, Path]:
    """Return the local poster index: filename stem (Chinese title) -> file path.

    Used to build a "featured movies" gallery straight from the `电影照片/`
    directory for the homepage.
    """
    return dict(_local_poster_index())


def local_poster_path(title: str, title_zh: str = "") -> Path | None:
    """Look up a local poster image by Chinese display title or English title.

    Returns None if the `电影照片` directory is missing or no matching file
    is found, so callers can fall back to TMDb or the gradient placeholder.
    """
    index = _local_poster_index()
    if not index:
        return None
    clean_title, _ = clean_title_year(title)
    for candidate in (title_zh, title, clean_title):
        candidate = candidate.strip() if candidate else ""
        if candidate and candidate in index:
            return index[candidate]
    return None


def fetch_poster_url(title: str, year: int | None = None, timeout: float = 3.0) -> str | None:
    """Look up a poster image URL on TMDb, or return None if unavailable.

    Returns None immediately (no network call) when `TMDB_API_KEY` is unset
    so the app works fully offline with gradient placeholders.
    """
    if not TMDB_API_KEY:
        return None
    clean_title, parsed_year = clean_title_year(title)
    params = {"api_key": TMDB_API_KEY, "query": clean_title}
    release_year = year or parsed_year
    if release_year:
        params["year"] = release_year
    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError):
        return None
    if not results:
        return None
    poster_path = results[0].get("poster_path")
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"
