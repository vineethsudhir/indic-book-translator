from pathlib import Path
from typing import Callable

from ..config import QAConfig, load_provider_config, resolve_api_key
from ..languages import LANGUAGES, TargetLanguage
from ..providers.factory import build_provider
from .backtranslate import LLMBackTranslator, LocalBackTranslator
from .base import BackTranslator, Embedder
from .embed import LocalMiniLMEmbedder, OpenAICompatibleEmbedder


def build_qa(
    cfg: QAConfig,
    *,
    default_provider_config_path: str,
    resolve_path: Callable[[str], Path],
    language: TargetLanguage = LANGUAGES["kn"],
) -> tuple[BackTranslator, Embedder] | None:
    """Build the configured back-translator and embedder.

    Returns None when QA is disabled, without importing any heavy backend.
    `resolve_path` maps a config-relative path string to an absolute Path (the
    caller passes the same `_resolve` used in scripts/translate_book.py).
    `language` is the target language whose FLORES code the local back-translator
    translates from and whose name the LLM prompt uses.
    """
    if not cfg.enabled:
        return None

    if cfg.back_translation == "indictrans2_local":
        back_translator: BackTranslator = LocalBackTranslator(
            resolve_path(cfg.indic_en_model_dir), target_flores=language.flores
        )
    elif cfg.back_translation == "llm":
        provider_config_path = cfg.llm_provider_config or default_provider_config_path
        provider_cfg = load_provider_config(resolve_path(provider_config_path))
        back_translator = LLMBackTranslator(
            build_provider(provider_cfg), language_name=language.name
        )
    else:
        raise ValueError(f"Unknown back_translation engine: {cfg.back_translation!r}")

    if cfg.embedding == "local_minilm":
        # Local models are Hugging Face repo ids ("org/name") or folders; a bare
        # name like "text-embedding-3-small" is an API model left over from
        # switching backends, and would fail with an opaque HF download error.
        if "/" not in cfg.embedding_model and not resolve_path(cfg.embedding_model).exists():
            raise ValueError(
                f"Embedding model {cfg.embedding_model!r} is an API model name, but the "
                "embedding backend is local_minilm. In Settings, set Embedding model to "
                "'sentence-transformers/all-MiniLM-L6-v2' (or switch the backend to "
                "openai_compatible)."
            )
        embedder: Embedder = LocalMiniLMEmbedder(cfg.embedding_model)
    elif cfg.embedding == "openai_compatible":
        if not cfg.embedding_base_url:
            raise ValueError(
                "'embedding_base_url' is required when embedding is 'openai_compatible'"
            )
        api_key = resolve_api_key(cfg.embedding_api_key_env or "OPENAI_API_KEY")
        embedder = OpenAICompatibleEmbedder(
            base_url=cfg.embedding_base_url,
            api_key=api_key,
            model=cfg.embedding_model,
        )
    else:
        raise ValueError(f"Unknown embedding engine: {cfg.embedding!r}")

    return back_translator, embedder
