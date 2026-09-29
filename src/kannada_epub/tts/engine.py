import re

import numpy as np

from .base import TTSProvider

# No trained Kannada sentence-segmentation model is wired in yet (see the
# analogous English-only note in translation/engine.py), so this splits on
# the punctuation Kannada prose actually uses in practice: ASCII . ! ? plus
# the Devanagari-derived danda marks । and ॥.
_KANNADA_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")

# Verified empirically (test_tts.py): a single generate() call tops out around
# ~30s of audio (max_length=2610 codec tokens) — text beyond that is silently
# truncated, not chunked, by the model itself. Grouping sentences into <=~200
# character pieces keeps every chunk well under that ceiling regardless of how
# long the source paragraph is.
_MAX_CHUNK_CHARS = 200

_EMOTION_PHRASES = {
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


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _KANNADA_SENTENCE_SPLIT_RE.split(text.strip())]
    return [s for s in sentences if s]


def _chunk_sentences(sentences: list[str], max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class IndicParlerTTSEngine(TTSProvider):
    """Wraps ai4bharat/indic-parler-tts for paragraph-level Kannada narration.

    Paragraphs are split into sentences and grouped into <=~200-character
    chunks (see _MAX_CHUNK_CHARS) before synthesis, since a single generate()
    call silently truncates text beyond ~30s of audio rather than erroring —
    the same "never silently drop content" concern that drove sentence
    chunking on the translation side.
    """

    def __init__(self, model_dir, device: str | None = None):
        # Imported here, not at module top, so that importing this module does
        # not require the optional local ML stack (torch/transformers/parler_tts).
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(str(model_dir)).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._description_tokenizer = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )
        self.sampling_rate = self._model.config.sampling_rate

    def _description_for(self, voice: str, emotion: str) -> str:
        emotion_phrase = _EMOTION_PHRASES.get(emotion, "a clear narration tone")
        return (
            f"{voice} speaks in {emotion_phrase} at a moderate pace, "
            f"close-sounding and high quality, with no background noise."
        )

    def _synthesize_chunk(self, text: str, voice: str, emotion: str) -> np.ndarray:
        description = self._description_for(voice, emotion)
        desc_ids = self._description_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").to(self.device)
        generation = self._model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
        audio = generation.to(self._torch.float32).cpu().numpy().squeeze()
        peak = np.abs(audio).max()
        # Per-chunk peak normalization so chunk boundaries within a paragraph
        # don't carry an audible volume jump — separate generate() calls have
        # no shared loudness reference.
        if peak > 1e-6:
            audio = audio / peak * 0.9
        return audio.astype(np.float32)

    def synthesize_paragraph(
        self, text: str, voice: str, emotion: str, intra_chunk_gap_s: float = 0.25
    ) -> np.ndarray:
        chunks = _chunk_sentences(_split_sentences(text)) or [text]
        gap = np.zeros(int(intra_chunk_gap_s * self.sampling_rate), dtype=np.float32)

        pieces: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                pieces.append(gap)
            pieces.append(self._synthesize_chunk(chunk, voice, emotion))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)