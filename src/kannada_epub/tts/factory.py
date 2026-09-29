from ..config import TTSModelConfig, resolve_api_key
from .base import TTSProvider
from .cloud import OpenAICompatibleTTSProvider, SarvamTTSProvider


def build_tts_provider(config: TTSModelConfig, local_model_dir: str | None = None) -> TTSProvider:
    if config.provider == "parler_local":
        if not local_model_dir:
            raise ValueError("'local_model_dir' is required for parler_local TTS")

        # Imported here, not at module top: the local engine needs the optional
        # ML stack (torch/transformers/parler_tts), absent from cloud-only
        # installs. Turn a missing dependency into an actionable error.
        try:
            from .engine import IndicParlerTTSEngine

            return IndicParlerTTSEngine(model_dir=local_model_dir)
        except ImportError as exc:
            raise RuntimeError(
                "TTS with provider 'parler_local' needs the local ML dependencies "
                "(torch, transformers, parler_tts). Install them with: "
                'pip install -e ".[local]"'
            ) from exc

    if config.provider == "sarvam":
        api_key = resolve_api_key(config.api_key_env or "SARVAM_API_KEY")
        return SarvamTTSProvider(
            api_key=api_key,
            voice=config.voice,
            model=config.model,
            language_code=config.language_code,
            base_url=config.base_url or "https://api.sarvam.ai",
            sampling_rate=config.sampling_rate,
        )

    if config.provider == "openai_compatible":
        if not config.base_url:
            raise ValueError("'base_url' is required when TTS provider is 'openai_compatible'")
        api_key = resolve_api_key(config.api_key_env or "OPENAI_API_KEY")
        return OpenAICompatibleTTSProvider(
            base_url=config.base_url,
            api_key=api_key,
            model=config.model,
            voice=config.voice,
            sampling_rate=config.sampling_rate,
        )

    raise ValueError(f"Unknown TTS provider: {config.provider!r}")
