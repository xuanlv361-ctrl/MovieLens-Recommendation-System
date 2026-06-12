"""Administrator authentication helpers.

Credentials are read from environment variables (optionally loaded from a
local `.env` file) so that no credentials are hardcoded in source files that
get committed to version control. See `.env.example` for the expected
variable names.
"""

from __future__ import annotations

import os

from .config import PROJECT_ROOT

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def get_admin_username() -> str:
    """Return the configured administrator username (env var or demo default)."""
    return os.environ.get("ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)


def get_admin_password() -> str:
    """Return the configured administrator password (env var or demo default)."""
    return os.environ.get("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)


def verify_admin(username: str, password: str) -> bool:
    """Check administrator credentials against the configured values."""
    return username == get_admin_username() and password == get_admin_password()
