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

# Helpers for `normalize_title`: strip a trailing "(YYYY)"/"（YYYY）" year,
# drop standalone "the"/"a"/"an" articles (handles MovieLens' "Title, The
# (Year)" form once the comma is replaced with a space), then strip every
# remaining character that isn't alphanumeric or CJK before lower-casing.
_YEAR_SUFFIX_RE = re.compile(r"[\(（]\s*\d{4}\s*[\)）]\s*$")
_ARTICLE_WORD_RE = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z一-鿿]+")

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


def normalize_title(text: str) -> str:
    """Normalize a movie title for loose matching.

    Strips a trailing "(YYYY)"/"（YYYY）" year, removes standalone "the"/"a"/
    "an" articles (so MovieLens' "Title, The (Year)" form lines up with
    "The Title", "Title The", and plain "Title" once the comma becomes a
    space), then drops every remaining non-alphanumeric/non-CJK character and
    lower-cases the result.
    """
    if not text:
        return ""
    text = _YEAR_SUFFIX_RE.sub("", text.strip())
    text = text.replace(",", " ")
    text = _ARTICLE_WORD_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub("", text)
    return text.lower()


# Strict alphanumeric-only normalization: lower-case, keep only a-z/0-9. Unlike
# `normalize_title`, this deliberately does NOT strip a trailing "(YYYY)" year
# or CJK characters, so a movie title like "Toy Story (1995)" normalizes to
# "toystory1995" -- matching a poster file renamed to "ToyStory1995.jpg" once
# all punctuation/spacing/parentheses were stripped from the original title.
_ALNUM_ONLY_RE = re.compile(r"[^a-z0-9]+")


def normalize_movie_name(name: str | None) -> str:
    """Lower-case `name` and strip every character that isn't a-z or 0-9."""
    if name is None:
        return ""
    return _ALNUM_ONLY_RE.sub("", str(name).lower())


def remove_year(title: str) -> str:
    """Strip a trailing "(YYYY)"/"（YYYY）" year from `title`, if present."""
    return _YEAR_SUFFIX_RE.sub("", str(title).strip())


@functools.lru_cache(maxsize=1)
def build_local_poster_index() -> dict[str, dict[str, Path]]:
    """Scan `电影照片/` and build exact-stem and normalized-title lookup maps.

    Returns a dict with three keys:
    - "exact": filename stem (without extension) -> file path
    - "normalized": `normalize_title(stem)` -> file path
    - "alnum": `normalize_movie_name(stem)` -> file path (year-preserving,
      alphanumeric-only -- see `normalize_movie_name`)

    Cached for the process lifetime; call `refresh_poster_cache()` after
    adding/removing poster files so the index is rebuilt.
    """
    exact: dict[str, Path] = {}
    normalized: dict[str, Path] = {}
    alnum: dict[str, Path] = {}
    if LOCAL_POSTER_DIR.is_dir():
        for path in sorted(LOCAL_POSTER_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in LOCAL_POSTER_EXTENSIONS:
                stem = path.stem.strip()
                exact.setdefault(stem, path)
                norm = normalize_title(stem)
                if norm:
                    normalized.setdefault(norm, path)
                key = normalize_movie_name(stem)
                if key:
                    alnum.setdefault(key, path)
    return {"exact": exact, "normalized": normalized, "alnum": alnum}


def build_movie_image_index(image_dir: str | Path = LOCAL_POSTER_DIR) -> dict[str, Path]:
    """Return the `normalize_movie_name(stem) -> file path` index for `image_dir`.

    For the default `电影照片/` directory this reuses the cached
    `build_local_poster_index()`. A different `image_dir` is scanned directly
    (uncached), which is mainly useful for tests/tools.
    """
    if Path(image_dir) == LOCAL_POSTER_DIR:
        return dict(build_local_poster_index()["alnum"])

    image_dir = Path(image_dir)
    index: dict[str, Path] = {}
    if image_dir.is_dir():
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in LOCAL_POSTER_EXTENSIONS:
                key = normalize_movie_name(path.stem.strip())
                if key:
                    index.setdefault(key, path)
    return index


def get_movie_image_path(movie_title: str, title_zh: str = "") -> Path | None:
    """Look up a poster for `movie_title` using strict alphanumeric matching.

    Tries, in order:
    1. Full standardized title (year included), e.g. "Toy Story (1995)" -> "toystory1995"
    2. The title with its trailing year removed, e.g. "toystory"
    3. A weak `contains`/`startswith` match against indexed filenames

    This is independent of `local_poster_path`'s Chinese-filename tiers and
    is used as a final fallback so renamed image files (special characters
    stripped) still resolve correctly.
    """
    index = build_movie_image_index()
    if not index:
        return None

    full_key = normalize_movie_name(movie_title)
    if full_key and full_key in index:
        return index[full_key]

    no_year_key = normalize_movie_name(remove_year(movie_title))
    if no_year_key and no_year_key in index:
        return index[no_year_key]

    zh_key = normalize_movie_name(title_zh)
    if zh_key and zh_key in index:
        return index[zh_key]

    for img_key, img_path in index.items():
        if full_key and (full_key in img_key or img_key in full_key):
            return img_path
        if no_year_key and (no_year_key in img_key or img_key in no_year_key):
            return img_path
    return None


def refresh_poster_cache() -> None:
    """Clear the cached local poster index so newly added files are picked up."""
    build_local_poster_index.cache_clear()


def list_local_posters() -> dict[str, Path]:
    """Return the local poster index: filename stem (Chinese title) -> file path.

    Used to build a "featured movies" gallery straight from the `电影照片/`
    directory for the homepage.
    """
    return dict(build_local_poster_index()["exact"])


def local_poster_path(title: str, title_zh: str = "") -> Path | None:
    """Look up a local poster image for a movie, trying several match tiers in order:

    1. `title_zh` exact filename match (e.g. "玩具总动员.jpg")
    2. `title` (original title with year) exact filename match (e.g. "Toy Story (1995).jpg")
    3. The title without its trailing year, exact filename match (e.g. "Toy Story.jpg")
    4. Loose normalized match, ignoring case/spacing/punctuation/brackets and
       MovieLens' "Title, The (Year)" article form (e.g. "教父.jpg" /
       "The Godfather.jpg" / "Godfather The.jpg" / "Godfather.jpg" all match
       "Godfather, The (1972)")
    5. Strict alphanumeric match via `get_movie_image_path` -- covers poster
       files renamed by stripping all special characters (year digits kept),
       e.g. "Toy Story (1995)" -> "toystory1995" matches "ToyStory1995.jpg"

    Returns None if the `电影照片` directory is missing or no tier matches, so
    callers can fall back to TMDb or the gradient placeholder.
    """
    index = build_local_poster_index()
    exact = index["exact"]
    if not exact:
        return None

    title = (title or "").strip()
    title_zh = (title_zh or "").strip()
    clean_title, _ = clean_title_year(title)

    if title_zh and title_zh in exact:
        return exact[title_zh]
    if title and title in exact:
        return exact[title]
    if clean_title and clean_title in exact:
        return exact[clean_title]

    normalized = index["normalized"]
    for candidate in (title_zh, title, clean_title):
        norm = normalize_title(candidate)
        if norm and norm in normalized:
            return normalized[norm]

    return get_movie_image_path(title, title_zh)


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
