import json
import re
from pathlib import Path

from .base import BackTranslator

_SYSTEM_PROMPT = """You are a Kannada-to-English translator. Translate each numbered Kannada paragraph below into literal, faithful English. Preserve meaning exactly: do not add, remove, summarize, or explain anything. Return ONLY a JSON array of translated strings, one per input paragraph, in the same order."""

# Same "first [...] block" fallback as
# src/kannada_epub/translation/cloud.py::_parse_output.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _numbered(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs, start=1))


def _parse_output(raw: str, expected_count: int) -> list[str]:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as err:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            raise RuntimeError(
                f"Back-translation did not return JSON. Raw output started with: {raw[:200]!r}"
            ) from err
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        got = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise RuntimeError(
            f"Back-translation returned {got} paragraphs for {expected_count} inputs — "
            f"refusing to misalign. Raw output started with: {raw[:200]!r}"
        )
    return [str(s) for s in parsed]


class LLMBackTranslator(BackTranslator):
    """Back-translate Kannada via an LLM chat provider.

    `provider` is any object exposing `complete(system_prompt, user_prompt)`
    (e.g. a ConsistencyEditorProvider). The response is parsed as a JSON array
    with a strict count check so paragraphs can never be misaligned.
    """

    def __init__(self, provider):
        self._provider = provider

    def back_translate(self, kannada: list[str]) -> list[str]:
        if not kannada:
            return []
        raw = self._provider.complete(_SYSTEM_PROMPT, _numbered(kannada))
        return _parse_output(raw, len(kannada))


class LocalBackTranslator(BackTranslator):
    """Back-translate Kannada with a local IndicTrans2 CTranslate2 model.

    `model_dir` is the converted CTranslate2 directory (the one containing the
    model files plus `vocab/model.SRC` and `vocab/model.TGT`). The heavy
    engine (and its ctranslate2/IndicTransToolkit imports) is loaded lazily in
    `__init__` so importing this module never requires the model or its deps.
    """

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"IndicTrans2 Kannada->English model not found at '{model_dir}'. "
                f"Run scripts/download_qa_models.py to download it."
            )

        # Imported here (not at module top) so nothing imports ctranslate2 /
        # IndicTransToolkit until a local back-translator is actually built.
        # Raise an actionable error when those optional deps are absent. The
        # construction is inside the same try because the engine imports its
        # heavy deps in __init__.
        try:
            from ..translation.engine import IndicTrans2Engine

            self._engine = IndicTrans2Engine(
                ct2_model_dir=model_dir,
                spm_src_path=model_dir / "vocab" / "model.SRC",
                spm_tgt_path=model_dir / "vocab" / "model.TGT",
                device="cpu",
                compute_type="int8",
            )
        except ImportError as exc:
            raise RuntimeError(
                "QA back-translation with 'indictrans2_local' needs the local ML "
                "dependencies (ctranslate2, sentencepiece, IndicTransToolkit, "
                "huggingface_hub, torch, transformers). Install them with: "
                'pip install -e ".[local]"'
            ) from exc

    def back_translate(self, kannada: list[str]) -> list[str]:
        if not kannada:
            return []
        return self._engine.translate_paragraphs(kannada, "kan_Knda", "eng_Latn")
