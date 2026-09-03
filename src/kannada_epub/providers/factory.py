from ..config import ProviderConfig, resolve_api_key
from .anthropic_provider import AnthropicProvider
from .base import ConsistencyEditorProvider
from .ollama_provider import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


def build_provider(config: ProviderConfig) -> ConsistencyEditorProvider:
    if config.provider == "ollama":
        return OllamaProvider(
            base_url=config.base_url or OLLAMA_DEFAULT_BASE_URL,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            think=config.think,
        )

    if config.provider == "openai_compatible":
        if not config.base_url:
            raise ValueError("'base_url' is required when provider is 'openai_compatible'")
        api_key = resolve_api_key(config.api_key_env or "OPENAI_API_KEY")
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "anthropic":
        api_key = resolve_api_key(config.api_key_env or "ANTHROPIC_API_KEY")
        return AnthropicProvider(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"Unknown provider: {config.provider!r}")