"""Offline tests for how cloud narration uses emotion tags (issue #21).

``httpx.post`` is faked to return a short WAV, so no network or API key is
needed. Sarvam gets a small ``pace`` change per emotion; OpenAI-compatible
speech gets ``instructions`` only for gpt-4o TTS models.

Run: .venv/bin/python scripts/test_tts_cloud.py
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from kannada_epub.config import TTSModelConfig  # noqa: E402
from kannada_epub.tts import cloud  # noqa: E402
from kannada_epub.tts.cloud import OpenAICompatibleTTSProvider, SarvamTTSProvider  # noqa: E402


def _wav_bytes(rate: int = 22050) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.full(rate // 10, 0.3, dtype=np.float32), rate, format="WAV")
    return buffer.getvalue()


class _Response:
    def __init__(self, *, json_body=None, content=b"", status_code=200):
        self._json, self.content, self.status_code = json_body, content, status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


def _capture(response: _Response) -> list[dict]:
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return response

    cloud.httpx.post = fake_post
    return calls


def check_sarvam() -> None:
    audio = {"audios": [base64.b64encode(_wav_bytes()).decode()]}
    for emotion, pace in (("Narration", None), ("Sad", 0.9), ("Anger", 1.1), ("Unknown", None)):
        calls = _capture(_Response(json_body=audio))
        SarvamTTSProvider(api_key="k").synthesize_paragraph("ನಮಸ್ಕಾರ.", "anushka", emotion)
        assert calls[0].get("pace") == pace, (emotion, calls[0])
        assert "pitch" not in calls[0] and "loudness" not in calls[0]
    calls = _capture(_Response(json_body=audio))
    SarvamTTSProvider(api_key="k", use_emotion=False).synthesize_paragraph("ನಮಸ್ಕಾರ.", "anushka", "Sad")
    assert "pace" not in calls[0], calls[0]
    for tag, value in cloud._SARVAM_EMOTION_PACE.items():
        assert 0.5 <= value <= 2.0, (tag, value)


def check_openai() -> None:
    wav = _Response(content=_wav_bytes(24000))
    calls = _capture(wav)
    OpenAICompatibleTTSProvider("https://x/v1", "k", "gpt-4o-mini-tts", "alloy").synthesize_paragraph(
        "ನಮಸ್ಕಾರ.", "alloy", "Sad"
    )
    assert "a sad tone" in calls[0]["instructions"], calls[0]
    for model in ("tts-1", "tts-1-hd", "some-other-tts"):
        calls = _capture(wav)
        OpenAICompatibleTTSProvider("https://x/v1", "k", model, "alloy").synthesize_paragraph(
            "ನಮಸ್ಕಾರ.", "alloy", "Sad"
        )
        assert "instructions" not in calls[0], (model, calls[0])
    calls = _capture(wav)
    OpenAICompatibleTTSProvider(
        "https://x/v1", "k", "gpt-4o-mini-tts", "alloy", use_emotion=False
    ).synthesize_paragraph("ನಮಸ್ಕಾರ.", "alloy", "Sad")
    assert "instructions" not in calls[0], calls[0]


def check_config() -> None:
    assert TTSModelConfig().use_emotion is True
    assert TTSModelConfig(use_emotion=False).use_emotion is False


def main() -> None:
    original = cloud.httpx.post
    try:
        check_sarvam()
        check_openai()
        check_config()
    finally:
        cloud.httpx.post = original
    print("test_tts_cloud: all assertions passed")


if __name__ == "__main__":
    main()
