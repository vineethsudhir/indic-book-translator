import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    provider: Literal["ollama", "openai_compatible", "anthropic"]
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    # Ollama-only: HTTP timeout in seconds. None = derive from max_tokens
    # (max(180, max_tokens / 8)); a fixed 180 s was too short for token-heavy
    # Kannada replies at the 4096 default.
    timeout_seconds: Optional[float] = None
    # Ollama-only: reasoning-capable models can burn the whole max_tokens
    # budget on hidden reasoning and return nothing. Off by default for
    # reliability; set true if you want the model to reason through edits
    # and don't mind occasional truncated (empty) results.
    think: bool = False


class TranslationModelConfig(BaseModel):
    provider: Literal["indictrans2_local", "openai_compatible"] = "indictrans2_local"
    model: str = "sarvam-m"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096


class TTSModelConfig(BaseModel):
    provider: Literal["parler_local", "sarvam", "openai_compatible"] = "parler_local"
    model: str = "bulbul:v3"
    voice: str = "anushka"
    language_code: str = "kn-IN"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    sampling_rate: int = 22050
    # Let cloud voices use each paragraph's emotion tag where they can
    # (Sarvam: pace; OpenAI gpt-4o TTS: instructions). Local Parler-TTS always
    # uses it.
    use_emotion: bool = True


class QAConfig(BaseModel):
    enabled: bool = False
    back_translation: Literal["indictrans2_local", "llm"] = "llm"
    indic_en_model_dir: str = "models/indic-en-200m-ct2/indic-en-200m-ct2/ctranslate2_model"
    llm_provider_config: Optional[str] = None  # path to a ProviderConfig YAML; None = reuse BookConfig.provider_config
    embedding: Literal["local_minilm", "openai_compatible"] = "local_minilm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_base_url: Optional[str] = None
    embedding_api_key_env: Optional[str] = None
    pass_threshold: float = 0.85
    flag_threshold: float = 0.70
    # On a QA retry, ask the translation engine to vary its decoding setup
    # (higher cloud temperature, or a wider local beam) instead of repeating
    # the call that already scored poorly. See TranslationProvider.
    vary_retry: bool = True


class BookConfig(BaseModel):
    epub_path: str
    output_dir: str
    glossary_db: str
    provider_config: str = "config/consistency_editor.example.yaml"
    ct2_model_dir: str = "models/en-indic-1b-ct2/en-indic-1b-ct2/ctranslate2_model"
    batch_size: int = 20
    context_tail_paragraphs: int = 2
    tone_register: str = "neutral, standard written Kannada"
    exclude_ids: list[str] = ["coverpage-wrapper"]
    strip_gutenberg: bool = False
    epubcheck: bool = False
    # FR-1.3: carry inline bold/italic/link markup through translation as
    # ⟦n⟧ … ⟦/n⟧ markers and rebuild it in the output EPUB. Off by default:
    # whether marker tokens hurt NMT quality is not measured yet.
    preserve_inline_markup: bool = False
    # Whether the consistency editor may see the previous chapter's English
    # tail as context. "auto" detects continuous novels from the chapter
    # titles; "carry"/"reset" force the choice. See detect_chapter_context.
    chapter_context: Literal["auto", "carry", "reset"] = "auto"
    limit_chapters: list[str] | None = None
    max_paragraphs_per_chapter: int | None = None
    translation: TranslationModelConfig = TranslationModelConfig()
    tts: TTSModelConfig = TTSModelConfig()
    qa: QAConfig = QAConfig()


def load_provider_config(path: str | Path) -> ProviderConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ProviderConfig.model_validate(raw["consistency_editor"])


def load_book_config(path: str | Path) -> BookConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return BookConfig.model_validate(raw["book"])


def resolve_api_key(env_var: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"Environment variable '{env_var}' is not set. "
            f"Set it (e.g. in a .env file or your shell) before using this provider."
        )
    return value
