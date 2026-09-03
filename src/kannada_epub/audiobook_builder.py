from pathlib import Path

import numpy as np
import soundfile as sf

from .book_translator import TranslatedBatch
from .tts import TTSProvider


def build_audiobook(
    batches: list[TranslatedBatch],
    voice: str,
    tts_engine: TTSProvider,
    output_path: str | Path,
    paragraph_gap_s: float = 0.6,
) -> Path:
    """Narrate every edited paragraph across `batches`, in order, as one
    continuous audiobook file. Each paragraph is synthesized with its own
    emotion tag (from the consistency-editing pass) and `voice`, then
    concatenated with a short silence between paragraphs."""
    gap = np.zeros(int(paragraph_gap_s * tts_engine.sampling_rate), dtype=np.float32)

    pieces: list[np.ndarray] = []
    total = sum(len(b.edited_kannada) for b in batches)
    done = 0
    for batch in batches:
        for text, emotion in zip(batch.edited_kannada, batch.edited_emotions):
            done += 1
            print(f"  [{done}/{total}] ({emotion}) {text[:40]}...")
            pieces.append(tts_engine.synthesize_paragraph(text, voice, emotion))
            pieces.append(gap)

    full_audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    output_path = Path(output_path)
    sf.write(str(output_path), full_audio, tts_engine.sampling_rate)
    return output_path