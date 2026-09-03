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


class BookConfig(BaseModel):
    epub_path: str
    output_dir: str
    glossary_db: str
    provider_config: str = "config/consistency_editor.example.yaml"
    ct2_model_dir: str = "models/en-indic-1b-ct2/en-indic-1b-ct2/ctranslate2_model"
    batch_size: int = 20
    context_tail_paragraphs: int = 2
    tone_register: str = "neutral, standard written Kannada"
    exclude_ids: list[str] = ["pg-header", "pg-footer", "coverpage-wrapper"]
    limit_chapters: list[str] | None = None
    max_paragraphs_per_chapter: int | None = None
    translation: TranslationModelConfig = TranslationModelConfig()
    tts: TTSModelConfig = TTSModelConfig()


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