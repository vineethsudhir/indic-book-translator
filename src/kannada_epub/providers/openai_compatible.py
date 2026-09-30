from openai import OpenAI

from .base import ConsistencyEditorProvider, OutputTruncatedError


class OpenAICompatibleProvider(ConsistencyEditorProvider):
    """Talks to any server implementing the OpenAI chat-completions API shape.

    This covers Ollama (which exposes an OpenAI-compatible endpoint alongside
    its native API), plus OpenAI itself, OpenRouter, Together, Groq, etc. —
    the only things that vary between them are base_url, api_key, and model.
    """

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        if response.choices[0].finish_reason == "length":
            raise OutputTruncatedError(
                "The model stopped generating because it hit max_tokens "
                f"({self._max_tokens}) — the reply is incomplete."
            )
        return response.choices[0].message.content or ""