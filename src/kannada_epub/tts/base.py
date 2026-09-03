from abc import ABC, abstractmethod

import numpy as np


class TTSProvider(ABC):
    """A paragraph-level Kannada narration backend.

    Implementations wrap a specific engine (local Indic Parler-TTS, Sarvam
    cloud Bulbul, OpenAI-compatible speech, ...). audiobook_builder only
    depends on this interface, so swapping engines never touches pipeline
    code. Cloud voices generally ignore the `emotion` tag (no equivalent
    parameter); implementations must document that.
    """

    sampling_rate: int

    @abstractmethod
    def synthesize_paragraph(self, text: str, voice: str, emotion: str) -> np.ndarray:
        """Narrate one paragraph as mono float32 audio at `sampling_rate`."""
        raise NotImplementedError
