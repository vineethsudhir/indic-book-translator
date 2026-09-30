"""App settings and secrets -- pure Python, no UI framework.

Everything the UI needs to persist lives here: the :class:`AppSettings`
pydantic model (cloud-first defaults), load/save to ``settings.yaml``, the
``secrets.env`` helpers (never logging or returning secret *values*), the
:func:`book_config_for` bridge into the pipeline's :class:`BookConfig`, and a
non-importing check for the optional local-ML libraries.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..config import (
    BookConfig,
    ProviderConfig,
    QAConfig,
    TranslationModelConfig,
    TTSModelConfig,
)
from ..epubcheck_runner import find_epubcheck
from .paths import (
    editor_config_path,
    glossary_db_path,
    outputs_dir,
    secrets_path,
    settings_path,
)

# Environment variables the UI offers a password box for. Any name matching
# ``_KEY_RE`` can still be saved via :func:`save_secret`.
KNOWN_KEYS = ["SARVAM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Modules that only exist in a ``pip install -e ".[local]"`` install. Checked
# with ``find_spec`` (never imported) so the cloud-only app works without them.
LOCAL_MODULES = [
    "torch",
    "transformers",
    "ctranslate2",
    "sentencepiece",
    "IndicTransToolkit",
    "parler_tts",
]


class AppSettings(BaseModel):
    """Everything a non-developer can tune, with cloud-first defaults."""

    translation: TranslationModelConfig = Field(
        default_factory=lambda: TranslationModelConfig(
            provider="openai_compatible",
            model="sarvam-m",
            base_url="https://api.sarvam.ai/v1",
            api_key_env="SARVAM_API_KEY",
        )
    )
    editor: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key_env="ANTHROPIC_API_KEY",
        )
    )
    tts: TTSModelConfig = Field(
        default_factory=lambda: TTSModelConfig(
            provider="sarvam",
            model="bulbul:v3",
            voice="anushka",
            api_key_env="SARVAM_API_KEY",
        )
    )
    qa: QAConfig = Field(
        default_factory=lambda: QAConfig(
            enabled=False,
            back_translation="llm",
            embedding="openai_compatible",
            embedding_model="text-embedding-3-small",
            embedding_base_url="https://api.openai.com/v1",
            embedding_api_key_env="OPENAI_API_KEY",
        )
    )
    batch_size: int = 20
    tone_register: str = "neutral, standard written Kannada"
    exclude_ids: list[str] = Field(
        default_factory=lambda: ["coverpage-wrapper"]
    )
    strip_gutenberg: bool = False
    # Validate the translated EPUB with EPUBCheck when it's installed.
    epubcheck: bool = True


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------
def load_settings() -> AppSettings:
    """Load settings, falling back to defaults when the file is missing."""
    path = settings_path()
    if not path.exists():
        return AppSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and raw.get("exclude_ids") == [
        "pg-header",
        "pg-footer",
        "coverpage-wrapper",
    ]:
        raw["exclude_ids"] = ["coverpage-wrapper"]
    return AppSettings.model_validate(raw)


def save_settings(settings: AppSettings) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            settings.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# secrets.env
# ---------------------------------------------------------------------------
def _read_secrets() -> dict[str, str]:
    path = secrets_path()
    if not path.exists():
        return {}
    secrets: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name:
            secrets[name] = value.strip()
    return secrets


def _write_secrets(secrets: dict[str, str]) -> None:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{name}={value}\n" for name, value in secrets.items())
    # Create/truncate with 0600 from the start; chmod again because os.open's
    # mode is masked by umask and an existing file keeps its old mode.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    if os.name == "posix":
        os.chmod(path, 0o600)


def save_secret(name: str, value: str) -> None:
    """Create/update a secret; an empty ``value`` deletes the key.

    Names must match ``^[A-Z][A-Z0-9_]*$`` and values must be single-line.
    Secret values are never logged or returned.
    """
    if not _KEY_RE.match(name):
        raise ValueError(
            f"Invalid secret name {name!r}; expected a shell-style name "
            f"matching ^[A-Z][A-Z0-9_]*$."
        )
    if "\n" in value or "\r" in value:
        raise ValueError("Secret values must not contain newlines.")

    secrets = _read_secrets()
    if value == "":
        secrets.pop(name, None)
    else:
        secrets[name] = value
    _write_secrets(secrets)


def load_secrets_into_env() -> None:
    """Export ``secrets.env`` into ``os.environ`` without overriding it."""
    for name, value in _read_secrets().items():
        if name not in os.environ:
            os.environ[name] = value


def secret_status() -> dict[str, bool]:
    """Which known keys are set (in ``secrets.env`` or the environment)."""
    file_secrets = _read_secrets()
    return {
        name: bool(file_secrets.get(name) or os.environ.get(name))
        for name in KNOWN_KEYS
    }


# ---------------------------------------------------------------------------
# Bridge to the pipeline
# ---------------------------------------------------------------------------
def book_config_for(epub_path: str | Path, settings: AppSettings) -> tuple[BookConfig, Path]:
    """Build a :class:`BookConfig` for one EPUB, writing the editor YAML.

    Returns the config and the per-book output directory
    (``outputs/<epub stem>``).
    """
    epub_path = Path(epub_path)
    output_dir = outputs_dir() / epub_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    editor_yaml = editor_config_path()
    editor_yaml.write_text(
        yaml.safe_dump(
            {"consistency_editor": settings.editor.model_dump(mode="json")},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    cfg = BookConfig(
        epub_path=str(epub_path),
        output_dir=str(output_dir),
        glossary_db=str(glossary_db_path()),
        provider_config=str(editor_yaml),
        batch_size=settings.batch_size,
        tone_register=settings.tone_register,
        exclude_ids=list(settings.exclude_ids),
        strip_gutenberg=settings.strip_gutenberg,
        epubcheck=settings.epubcheck,
        translation=settings.translation,
        tts=settings.tts,
        qa=settings.qa,
    )
    return cfg, output_dir


# ---------------------------------------------------------------------------
# Local-mode availability
# ---------------------------------------------------------------------------
def local_mode_status() -> dict[str, bool]:
    """Whether each optional local-ML library is importable (never imported)."""
    status: dict[str, bool] = {}
    for name in LOCAL_MODULES:
        try:
            status[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            status[name] = False
    return status


# ---------------------------------------------------------------------------
# Optional EPUBCheck
# ---------------------------------------------------------------------------
def epubcheck_status() -> dict:
    """Whether the EPUBCheck validator can run, for ``GET /api/state``.

    Only the jar's file name is exposed, never the user's home path.
    """
    found = find_epubcheck()
    return {
        "available": found is not None,
        "jar": found[1].name if found is not None else None,
        "how_to": (
            "Put epubcheck.jar (from the W3C EPUBCheck releases) in "
            "<data directory>/epubcheck/ and install Java."
        ),
    }
