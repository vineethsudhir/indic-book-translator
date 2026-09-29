from .base import TTSProvider
from .cloud import OpenAICompatibleTTSProvider, SarvamTTSProvider
from .factory import build_tts_provider

__all__ = [
    "TTSProvider",
    "IndicParlerTTSEngine",
    "SarvamTTSProvider",
    "OpenAICompatibleTTSProvider",
    "build_tts_provider",
]


def __getattr__(name: str):
    """Import the local Parler-TTS engine lazily.

    ``IndicParlerTTSEngine`` lives in ``.engine``, which needs the optional
    local ML stack (torch/transformers/parler_tts). Keeping it out of the
    module import means ``kannada_epub.tts`` imports fine in a cloud-only
    install. ``from kannada_epub.tts import IndicParlerTTSEngine`` still works
    via this hook on first attribute access.
    """
    if name == "IndicParlerTTSEngine":
        from .engine import IndicParlerTTSEngine

        return IndicParlerTTSEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
