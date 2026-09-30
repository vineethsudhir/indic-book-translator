"""Offline unit tests for the consistency-editor providers.

Covers truncation detection (Ollama `done_reason == "length"`, OpenAI
`finish_reason == "length"`, Anthropic `stop_reason == "max_tokens"`) and
Ollama's HTTP timeout, which scales with the token budget when no explicit
value is given. `httpx.post` is monkeypatched and the SDK clients are replaced
with fakes, so no network is touched.

Run: .venv/bin/python scripts/test_providers.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import kannada_epub.providers.ollama_provider as ollama_provider
import kannada_epub.translation.cloud as translation_cloud
from kannada_epub.providers.anthropic_provider import AnthropicProvider
from kannada_epub.providers.base import OutputTruncatedError
from kannada_epub.providers.ollama_provider import OllamaProvider
from kannada_epub.providers.openai_compatible import OpenAICompatibleProvider
from kannada_epub.translation.cloud import OpenAICompatibleTranslationProvider
from kannada_epub.translation.engine import IndicTrans2Engine


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _install_fake_post(payload: dict, calls: list) -> None:
    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(payload)

    ollama_provider.httpx.post = fake_post


def _check_ollama() -> None:
    original_post = ollama_provider.httpx.post
    try:
        # Normal stop: returns the content and uses the budget-scaled timeout.
        calls: list[dict] = []
        _install_fake_post(
            {"message": {"content": "edited text"}, "done_reason": "stop"}, calls
        )
        provider = OllamaProvider("http://localhost:11434", "gemma", 0.2, 4096)
        assert provider.complete("system", "user") == "edited text"
        assert calls[-1]["timeout"] == max(180.0, 4096 / 8) == 512.0
        assert calls[-1]["url"] == "http://localhost:11434/api/chat"
        assert calls[-1]["json"]["options"]["num_predict"] == 4096

        # A small budget keeps the 180 s floor.
        _install_fake_post({"message": {"content": "ok"}, "done_reason": "stop"}, calls)
        OllamaProvider("http://localhost:11434", "gemma", 0.2, 100).complete("s", "u")
        assert calls[-1]["timeout"] == 180.0

        # An explicit timeout_seconds wins over the derived value.
        _install_fake_post({"message": {"content": "ok"}, "done_reason": "stop"}, calls)
        OllamaProvider(
            "http://localhost:11434", "gemma", 0.2, 4096, timeout_seconds=42
        ).complete("s", "u")
        assert calls[-1]["timeout"] == 42

        # done_reason == "length" means the reply was cut off.
        _install_fake_post(
            {"message": {"content": "cut off"}, "done_reason": "length"}, calls
        )
        try:
            OllamaProvider("http://localhost:11434", "gemma", 0.2, 4096).complete("s", "u")
        except OutputTruncatedError:
            pass
        else:
            raise AssertionError("expected OutputTruncatedError for done_reason=length")
    finally:
        ollama_provider.httpx.post = original_post


def _make_openai_client(content, finish_reason):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )

    def create(**_kwargs):
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _check_openai() -> None:
    provider = OpenAICompatibleProvider(
        "http://example.invalid/v1", "fake-key", "gpt", 0.2, 4096
    )
    provider._client = _make_openai_client("edited text", "stop")
    assert provider.complete("system", "user") == "edited text"

    provider._client = _make_openai_client(None, "stop")
    assert provider.complete("system", "user") == ""  # None content stays empty

    provider._client = _make_openai_client("cut off", "length")
    try:
        provider.complete("system", "user")
    except OutputTruncatedError:
        pass
    else:
        raise AssertionError("expected OutputTruncatedError for finish_reason=length")


def _make_anthropic_client(text, stop_reason):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )

    def create(**_kwargs):
        return response

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _check_anthropic() -> None:
    provider = AnthropicProvider("fake-key", "claude", 0.2, 4096)
    provider._client = _make_anthropic_client("edited text", "end_turn")
    assert provider.complete("system", "user") == "edited text"

    provider._client = _make_anthropic_client("cut off", "max_tokens")
    try:
        provider.complete("system", "user")
    except OutputTruncatedError:
        pass
    else:
        raise AssertionError("expected OutputTruncatedError for stop_reason=max_tokens")


def _install_fake_translation_post(payload: dict, calls: list) -> None:
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(payload)

    translation_cloud.httpx.post = fake_post


def _check_cloud_translation_retry() -> None:
    """The cloud retry raises temperature + adds the faithfulness instruction."""
    original_post = translation_cloud.httpx.post
    calls: list[dict] = []
    try:
        provider = OpenAICompatibleTranslationProvider(
            "http://example.invalid/v1", "fake-key", "gpt", temperature=0.2
        )

        # Normal call: base temperature, plain system prompt.
        _install_fake_translation_post(
            {"choices": [{"message": {"content": '["ಎರಡು"]'}}]}, calls
        )
        assert provider.translate_paragraphs(["two"], "eng_Latn", "kan_Knda") == ["ಎರಡು"]
        normal = calls[-1]["json"]
        assert normal["temperature"] == 0.2, normal["temperature"]
        assert normal["messages"][0]["content"] == translation_cloud._SYSTEM_PROMPT
        assert normal["messages"][1]["content"].startswith("[1] two")

        # Retry: temperature + 0.4 and the extra instruction appended.
        _install_fake_translation_post(
            {"choices": [{"message": {"content": '["ಎರಡು"]'}}]}, calls
        )
        assert provider.translate_paragraphs_retry(["two"], "eng_Latn", "kan_Knda") == ["ಎರಡು"]
        retry = calls[-1]["json"]
        assert abs(retry["temperature"] - 0.6) < 1e-9, retry["temperature"]
        prompt = retry["messages"][0]["content"]
        assert prompt.startswith(translation_cloud._SYSTEM_PROMPT)
        assert prompt != translation_cloud._SYSTEM_PROMPT
        assert "faithful" in prompt and "transliterating" in prompt
        assert provider.retry_description == "higher temperature, faithfulness instruction"

        # The raised temperature is capped at 1.0.
        hot = OpenAICompatibleTranslationProvider(
            "http://example.invalid/v1", "fake-key", "gpt", temperature=0.9
        )
        _install_fake_translation_post(
            {"choices": [{"message": {"content": '["ಒಂದು"]'}}]}, calls
        )
        hot.translate_paragraphs_retry(["one"], "eng_Latn", "kan_Knda")
        assert calls[-1]["json"]["temperature"] == 1.0, calls[-1]["json"]["temperature"]

        # A wrong count in a retry raises, exactly like the normal path.
        _install_fake_translation_post(
            {"choices": [{"message": {"content": '["only one"]'}}]}, calls
        )
        try:
            provider.translate_paragraphs_retry(["a", "b"], "eng_Latn", "kan_Knda")
        except RuntimeError as exc:
            assert "refusing to misalign" in str(exc), exc
        else:
            raise AssertionError("expected RuntimeError for a misaligned retry")
    finally:
        translation_cloud.httpx.post = original_post


class _FakeProcessor:
    def preprocess_batch(self, sentences, src_lang, tgt_lang, is_target=True):
        return list(sentences)

    def postprocess_batch(self, merged, lang=None):
        return list(merged)


class _FakeSentencePiece:
    def encode(self, text, out_type=str):
        return text.split()


class _FakeTranslator:
    def __init__(self):
        self.calls: list[dict] = []

    def translate_batch(self, tokenized, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(hypotheses=[["▁" + " ".join(tokenized[i][2:])]])
            for i in range(len(tokenized))
        ]


def _check_local_engine_retry() -> None:
    """The local retry widens the beam and adds a repetition penalty.

    Built with ``object.__new__`` and fake dependencies so the test needs no
    models and passes without ctranslate2/sentencepiece installed (the engine
    module imports them only inside methods).
    """
    engine = object.__new__(IndicTrans2Engine)
    engine._processor = _FakeProcessor()
    engine._sp_src = _FakeSentencePiece()
    engine._sp_tgt = _FakeSentencePiece()
    engine._translator = _FakeTranslator()
    engine._beam_size = 5
    engine._max_content_tokens = 200
    engine._max_input_length = 208
    engine._max_decoding_length = 256

    # Normal path: configured beam, no repetition penalty.
    engine.translate(["one two"], "eng_Latn", "kan_Knda")
    assert engine._translator.calls[-1]["beam_size"] == 5, engine._translator.calls[-1]
    assert "repetition_penalty" not in engine._translator.calls[-1]

    # Overrides threaded through translate().
    engine.translate(
        ["one two"], "eng_Latn", "kan_Knda", beam_size=8, repetition_penalty=1.2
    )
    assert engine._translator.calls[-1]["beam_size"] == 8
    assert engine._translator.calls[-1]["repetition_penalty"] == 1.2

    # Paragraph retry uses the wider beam (max(5 + 3, 8)) + penalty.
    engine.translate_paragraphs_retry(["one two"], "eng_Latn", "kan_Knda")
    assert engine._translator.calls[-1]["beam_size"] == 8
    assert engine._translator.calls[-1]["repetition_penalty"] == 1.2
    assert engine.retry_description == "wider beam, repetition penalty"

    # The paragraph-level normal call is untouched.
    engine.translate_paragraphs(["one two"], "eng_Latn", "kan_Knda")
    assert engine._translator.calls[-1]["beam_size"] == 5
    assert "repetition_penalty" not in engine._translator.calls[-1]


def main() -> None:
    _check_ollama()
    _check_openai()
    _check_anthropic()
    _check_cloud_translation_retry()
    _check_local_engine_retry()
    print("test_providers: all assertions passed")


if __name__ == "__main__":
    main()
