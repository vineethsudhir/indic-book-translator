import json
import re

import httpx

from .base import TranslationProvider

_SYSTEM_PROMPT_TEMPLATE = """You are an English-to-{name} literary translator. Translate each numbered paragraph below into natural, standard written {name}. Preserve meaning exactly: do not add, remove, or explain anything. Keep proper nouns, acronyms, and code in their original form. Return ONLY a JSON array of translated strings, one per input paragraph, in the same order."""

# Appended to the system prompt on a QA retry. The retry exists because the
# first attempt scored low, so it asks for the two failure modes QA catches
# most: dropped detail and English words left transliterated (not translated).
_RETRY_INSTRUCTION_TEMPLATE = (
    "This is a second attempt at the same passage: produce a faithful, "
    "complete translation that keeps every detail of the original, and "
    "translate ordinary English words into {name} rather than "
    "transliterating them."
)


def system_prompt_for(language_name: str) -> str:
    """The literary translator's system prompt for a target language name."""
    return _SYSTEM_PROMPT_TEMPLATE.format(name=language_name)


def retry_instruction_for(language_name: str) -> str:
    """The retry instruction for a target language name."""
    return _RETRY_INSTRUCTION_TEMPLATE.format(name=language_name)


# Kannada forms, kept for callers/tests that reference the constants.
_SYSTEM_PROMPT = system_prompt_for("Kannada")
_RETRY_INSTRUCTION = retry_instruction_for("Kannada")


def _numbered(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs, start=1))


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_output(raw: str, expected_count: int) -> list[str]:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as err:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            raise RuntimeError(f"Cloud translation did not return JSON. Raw output started with: {raw[:200]!r}") from err
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        got = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise RuntimeError(
            f"Cloud translation returned {got} paragraphs for {expected_count} inputs — "
            f"refusing to misalign. Raw output started with: {raw[:200]!r}"
        )
    return [str(s) for s in parsed]


class OpenAICompatibleTranslationProvider(TranslationProvider):
    """Translate via any OpenAI-compatible chat-completions API.

    One call per batch of paragraphs (chunked by character budget), with a
    strict paragraph-count check on every response — the same "never
    misalign" guarantee as the local engine. Verified shape against Sarvam's
    /v1/chat/completions (Bearer auth accepted); also fits OpenAI, OpenRouter,
    Together, Groq, and local OpenAI-compatible servers.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_chars_per_call: int = 6000,
        language_name: str = "Kannada",
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_chars_per_call = max_chars_per_call
        self._system_prompt = system_prompt_for(language_name)
        self._retry_instruction = retry_instruction_for(language_name)

    def _translate_chunk(
        self, chunk: list[str], *, temperature: float, system_prompt: str
    ) -> list[str]:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": _numbered(chunk)},
                        ],
                        "temperature": temperature,
                        "max_tokens": self._max_tokens,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
                return _parse_output(raw, len(chunk))
            except RuntimeError as e:
                last_error = e
        raise last_error  # type: ignore[misc]

    def _translate_all(
        self,
        paragraphs: list[str],
        *,
        temperature: float,
        system_prompt: str,
    ) -> list[str]:
        if not paragraphs:
            return []
        out: list[str] = []
        chunk: list[str] = []
        chunk_chars = 0
        for p in paragraphs:
            if chunk and chunk_chars + len(p) > self._max_chars_per_call:
                out.extend(
                    self._translate_chunk(
                        chunk, temperature=temperature, system_prompt=system_prompt
                    )
                )
                chunk, chunk_chars = [], 0
            chunk.append(p)
            chunk_chars += len(p)
        if chunk:
            out.extend(
                self._translate_chunk(
                    chunk, temperature=temperature, system_prompt=system_prompt
                )
            )
        return out

    def translate_paragraphs(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        return self._translate_all(
            paragraphs,
            temperature=self._temperature,
            system_prompt=self._system_prompt,
        )

    def translate_paragraphs_retry(
        self, paragraphs: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Retry with a raised temperature and a faithfulness instruction.

        The same chunking/parsing/count checks as the normal call apply; only
        the sampling temperature and system prompt differ, so a retry cannot
        silently shift paragraphs onto the wrong positions.
        """
        return self._translate_all(
            paragraphs,
            temperature=min(self._temperature + 0.4, 1.0),
            system_prompt=f"{self._system_prompt} {self._retry_instruction}",
        )

    retry_description = "higher temperature, faithfulness instruction"
