"""Unit test for IndicParlerTTSEngine's chunk handling, with generation
stubbed out — no model, no torch. Covers the sampled-generation failure
where Parler returns a single sample (shape (1, 1)) for a short prompt."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kannada_epub.tts.engine import IndicParlerTTSEngine


def stub_engine(outputs: list[np.ndarray]) -> tuple[IndicParlerTTSEngine, list[int]]:
    engine = object.__new__(IndicParlerTTSEngine)
    engine.sampling_rate = 1000
    calls = [0]

    def fake_generate(text, voice, emotion):
        out = outputs[min(calls[0], len(outputs) - 1)]
        calls[0] += 1
        return np.asarray(out, dtype=np.float32).reshape(-1)

    engine._generate = fake_generate
    return engine, calls


speech = np.full(500, 0.5, dtype=np.float32)
one_sample = np.zeros((1, 1), dtype=np.float32)

# A (1, 1) generation is retried, and the retry's audio is used.
engine, calls = stub_engine([one_sample, speech])
audio = engine._synthesize_chunk("I.", "Chetan", "Narration")
assert audio.ndim == 1 and audio.size == 500, audio.shape
assert calls[0] == 2
assert abs(float(np.abs(audio).max()) - 0.9) < 1e-6  # peak-normalized

# If every attempt is near-empty, the result is still 1-D and joinable.
engine, calls = stub_engine([one_sample])
audio = engine._synthesize_chunk("I.", "Chetan", "Narration")
assert audio.ndim == 1 and calls[0] == 3
assert np.concatenate([audio, np.zeros(10, dtype=np.float32)]).ndim == 1

# Normal output is used on the first attempt.
engine, calls = stub_engine([speech])
engine._synthesize_chunk("ನಾನು ಬಂದೆ.", "Chetan", "Narration")
assert calls[0] == 1

print("test_tts_engine: all assertions passed")
