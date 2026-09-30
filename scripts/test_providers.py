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
from kannada_epub.providers.anthropic_provider import AnthropicProvider
from kannada_epub.providers.base import OutputTruncatedError
from kannada_epub.providers.ollama_provider import OllamaProvider
from kannada_epub.providers.openai_compatible import OpenAICompatibleProvider


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


def main() -> None:
    _check_ollama()
    _check_openai()
    _check_anthropic()
    print("test_providers: all assertions passed")


if __name__ == "__main__":
    main()
