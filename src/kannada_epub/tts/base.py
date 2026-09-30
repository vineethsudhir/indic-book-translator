from abc import ABC, abstractmethod

import numpy as np

# How each consistency-editor emotion tag (consistency_editor.EMOTIONS) is
# described to a voice that takes a free-text style: Parler-TTS descriptions
# and OpenAI gpt-4o TTS instructions share this wording.
EMOTION_PHRASES = {
    "Command": "a commanding tone",
    "Anger": "an angry tone",
    "Narration": "a clear narration tone",
    "Conversation": "a conversational tone",
    "Disgust": "a disgusted tone",
    "Fear": "a fearful tone",
    "Happy": "a happy tone",
    "Neutral": "a neutral tone",
    "Proper Noun": "a clear, deliberate tone",
    "News": "a news-reading tone",
    "Sad": "a sad tone",
    "Surprise": "a surprised tone",
}


class TTSProvider(ABC):
    """A paragraph-level Kannada narration backend.

    Implementations wrap a specific engine (local Indic Parler-TTS, Sarvam
    cloud Bulbul, OpenAI-compatible speech, ...). audiobook_builder only
    depends on this interface, so swapping engines never touches pipeline
    code. Each implementation documents how it uses the `emotion` tag: local
    Parler-TTS describes it, Sarvam maps it to a small pace change, and
    OpenAI-compatible speech sends it as instructions to gpt-4o TTS models
    only.
    """

    sampling_rate: int

    @abstractmethod
    def synthesize_paragraph(self, text: str, voice: str, emotion: str) -> np.ndarray:
        """Narrate one paragraph as mono float32 audio at `sampling_rate`."""
        raise NotImplementedError
