from .base import TranslationProvider
from .cloud import OpenAICompatibleTranslationProvider
from .factory import build_translation_provider

__all__ = [
    "TranslationProvider",
    "IndicTrans2Engine",
    "OpenAICompatibleTranslationProvider",
    "build_translation_provider",
]


def __getattr__(name: str):
    """Import the local IndicTrans2 engine lazily.

    ``IndicTrans2Engine`` lives in ``.engine``, which needs the optional local
    ML stack (ctranslate2/sentencepiece/IndicTransToolkit). Keeping it out of
    the module import means ``kannada_epub.translation`` — and everything built
    on it — imports fine in a cloud-only install. ``from
    kannada_epub.translation import IndicTrans2Engine`` still works: the import
    system falls back to this hook on first attribute access.
    """
    if name == "IndicTrans2Engine":
        from .engine import IndicTrans2Engine

        return IndicTrans2Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
