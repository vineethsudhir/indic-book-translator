"""Per-user data directory and file locations for the app.

All user state (settings, secrets, uploaded books, translation outputs and the
glossary DB) lives under a single per-user data directory chosen by the OS --
never inside the repository, which will be read-only once this is packaged as a
desktop app. ``KANNADA_APP_DATA_DIR`` overrides the location (tests use a temp
directory).
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "KannadaBookTranslator"


def data_dir() -> Path:
    """Return (and create) the per-user data directory."""
    override = os.environ.get("KANNADA_APP_DATA_DIR")
    base = (
        Path(override)
        if override
        else Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def settings_path() -> Path:
    return data_dir() / "settings.yaml"


def secrets_path() -> Path:
    return data_dir() / "secrets.env"


def editor_config_path() -> Path:
    return data_dir() / "editor.yaml"


def books_dir() -> Path:
    path = data_dir() / "books"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir() -> Path:
    path = data_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def glossary_db_path() -> Path:
    return data_dir() / "glossary.db"
