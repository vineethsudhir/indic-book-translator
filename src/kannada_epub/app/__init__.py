"""Local web app for the translator: ``python -m kannada_epub.app``.

Runs the translation pipeline in-process (no subprocess) and keeps all user
state in a per-user data directory; works with none of the optional local-ML
libraries installed.
"""

from .paths import data_dir

__all__ = ["data_dir"]
