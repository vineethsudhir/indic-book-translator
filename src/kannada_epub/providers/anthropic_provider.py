from anthropic import Anthropic

from .base import ConsistencyEditorProvider, OutputTruncatedError


class AnthropicProvider(ConsistencyEditorProvider):
    """Native Anthropic Messages API adapter (schema differs from OpenAI's)."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        if response.stop_reason == "max_tokens":
            raise OutputTruncatedError(
                "Anthropic stopped generating because it hit max_tokens "
                f"({self._max_tokens}) — the reply is incomplete."
            )
        return "".join(block.text for block in response.content if block.type == "text")