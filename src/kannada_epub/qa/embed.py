import httpx

from .base import Embedder

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_TOKENS = 256


class LocalMiniLMEmbedder(Embedder):
    """Sentence embeddings from a local `transformers` model (no extra dep).

    Uses AutoTokenizer/AutoModel with mean pooling over the attention mask and
    L2 normalization — the standard Sentence-Transformers recipe for
    `all-MiniLM-L6-v2`, reproduced so we do not add a sentence-transformers
    dependency. transformers/torch are imported lazily in `__init__`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "QA embedding with 'local_minilm' needs the local ML dependencies "
                "(torch, transformers). Install them with: "
                'pip install -e ".[local]"'
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._batch_size = batch_size
        self._max_tokens = max_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        torch = self._torch
        vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_tokens,
                    return_tensors="pt",
                )
                output = self._model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"]

                # Mean pooling over the attention mask.
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts

                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                vectors.extend(normalized.tolist())
        return vectors


class OpenAICompatibleEmbedder(Embedder):
    """Embeddings from any OpenAI-compatible `POST {base_url}/embeddings` API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(texts):
            raise RuntimeError(
                f"Embedding API returned {len(data)} vectors for {len(texts)} inputs — "
                f"refusing to misalign."
            )
        # Some servers return results out of order; index is authoritative.
        ordered = sorted(data, key=lambda item: item["index"])
        return [list(item["embedding"]) for item in ordered]
