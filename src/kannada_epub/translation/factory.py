from ..config import TranslationModelConfig, resolve_api_key
from .base import TranslationProvider
from .cloud import OpenAICompatibleTranslationProvider


def build_translation_provider(
    config: TranslationModelConfig,
    ct2_model_dir: str | None = None,
    language_name: str = "Kannada",
) -> TranslationProvider:
    if config.provider == "indictrans2_local":
        if not ct2_model_dir:
            raise ValueError("'ct2_model_dir' is required for indictrans2_local translation")
        from pathlib import Path

        # Imported here, not at module top: the local engine pulls in the
        # optional ML stack (ctranslate2/sentencepiece/IndicTransToolkit), which
        # a cloud-only install does not have. Convert a missing dependency into
        # an actionable error instead of a bare ImportError.
        try:
            from .engine import IndicTrans2Engine

            model_dir = Path(ct2_model_dir)
            return IndicTrans2Engine(
                ct2_model_dir=model_dir,
                spm_src_path=model_dir / "vocab" / "model.SRC",
                spm_tgt_path=model_dir / "vocab" / "model.TGT",
                device="cpu",
                compute_type="int8",
            )
        except ImportError as exc:
            raise RuntimeError(
                "Translation with provider 'indictrans2_local' needs the local ML "
                "dependencies (ctranslate2, sentencepiece, IndicTransToolkit, "
                "huggingface_hub, torch, transformers). Install them with: "
                'pip install -e ".[local]"'
            ) from exc

    if config.provider == "openai_compatible":
        if not config.base_url:
            raise ValueError("'base_url' is required when translation provider is 'openai_compatible'")
        api_key = resolve_api_key(config.api_key_env or "OPENAI_API_KEY")
        return OpenAICompatibleTranslationProvider(
            base_url=config.base_url,
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            language_name=language_name,
        )

    raise ValueError(f"Unknown translation provider: {config.provider!r}")
