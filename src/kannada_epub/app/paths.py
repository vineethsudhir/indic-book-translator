"""Per-user file locations for the app. Nothing lives inside the repository,
which will be read-only once this is packaged as a desktop app.

Two places, split by who needs to find the files:
- App-private state (settings, API keys, glossary DB) goes in the OS's app
  data directory (e.g. ~/Library/Application Support on macOS), where users
  don't browse and keys stay out of the way.
- User-facing files (uploaded books and translation outputs) go in
  Documents/KannadaBookTranslator, where people expect to find their work.

``KANNADA_DOCUMENTS_DIR`` overrides the Documents location.
``KANNADA_APP_DATA_DIR`` overrides the data directory and, unless
``KANNADA_DOCUMENTS_DIR`` is also set, puts books/outputs under it too, so
tests stay inside one temp directory.
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


def documents_dir() -> Path:
    """Return (and create) the user-visible folder for books and outputs."""
    override = os.environ.get("KANNADA_DOCUMENTS_DIR")
    if override:
        base = Path(override)
    elif os.environ.get("KANNADA_APP_DATA_DIR"):
        base = data_dir()
    else:
        base = Path(platformdirs.user_documents_dir()) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def books_dir() -> Path:
    path = documents_dir() / "books"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir() -> Path:
    path = documents_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sites_dir() -> Path:
    """Return (and create) the folder holding exported static websites."""
    path = documents_dir() / "sites"
    path.mkdir(parents=True, exist_ok=True)
    return path


def glossary_db_path() -> Path:
    return data_dir() / "glossary.db"
