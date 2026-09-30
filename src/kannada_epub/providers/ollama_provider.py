from typing import Optional

import httpx

from .base import ConsistencyEditorProvider, OutputTruncatedError


class OllamaProvider(ConsistencyEditorProvider):
    """Talks to Ollama's native /api/chat endpoint (not the OpenAI-compat shim).

    Reasoning-capable Ollama models (e.g. gemma4) can spend their entire
    max_tokens budget on hidden reasoning and return empty content — this is
    reproducible, not hypothetical (verified: ~1 in 4 calls on gemma4:26b at
    max_tokens=4096). The OpenAI-compat endpoint silently drops the `think`
    field; only the native API honors it, which removes the reasoning phase
    entirely instead of just hoping it finishes in time.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        think: bool = False,
        timeout_seconds: Optional[float] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._think = think
        self._timeout_seconds = timeout_seconds

    @property
    def _timeout(self) -> float:
        """HTTP timeout: an explicit value, else scale with the token budget.

        Kannada is token-heavy, so a max_tokens-sized reply can take far longer
        than the old hard-coded 180 s. ``max_tokens / 8`` gives ~512 s at the
        4096 default, with the 180 s floor kept for small budgets.
        """
        if self._timeout_seconds is not None:
            return self._timeout_seconds
        return max(180.0, self._max_tokens / 8)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "think": self._think,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("done_reason") == "length":
            raise OutputTruncatedError(
                "Ollama stopped generating because it hit num_predict "
                f"({self._max_tokens} tokens) — the reply is incomplete."
            )
        return payload["message"]["content"]