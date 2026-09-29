from abc import ABC, abstractmethod


class BackTranslator(ABC):
    """Translates a batch of Kannada strings back into English.

    Implementations must return exactly one output per input, in the same
    order. A backend that cannot honour that (wrong count, parse failure)
    must raise rather than return a misaligned list.
    """

    @abstractmethod
    def back_translate(self, kannada: list[str]) -> list[str]:
        raise NotImplementedError


class Embedder(ABC):
    """Turns a batch of texts into fixed-size embedding vectors.

    Implementations must return exactly one vector per input, in the same
    order. A backend that cannot honour that must raise rather than return a
    misaligned list.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
