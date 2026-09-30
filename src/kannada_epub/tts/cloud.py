import base64
import io
import re

import httpx
import numpy as np
import soundfile as sf

from .base import EMOTION_PHRASES, TTSProvider

_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def _decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    target_len = max(1, int(duration * dst_rate))
    old_index = np.linspace(0, len(audio) - 1, num=len(audio))
    new_index = np.linspace(0, len(audio) - 1, num=target_len)
    return np.interp(new_index, old_index, audio).astype(np.float32)


def _split_for_limit(text: str, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", text.strip()) if s.strip()]
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
    return chunks or [text]


# Sarvam Bulbul has no emotion parameter; v3 dropped pitch and loudness and
# keeps pace (0.5-2.0). Emotion becomes a small pace change; tags not listed
# keep the default pace, so the request is unchanged for them.
_SARVAM_EMOTION_PACE = {
    "Sad": 0.9,
    "Fear": 1.05,
    "Happy": 1.05,
    "Command": 1.05,
    "Anger": 1.1,
    "Surprise": 1.1,
}


def _sarvam_pace(emotion: str) -> float:
    return min(max(_SARVAM_EMOTION_PACE.get(emotion, 1.0), 0.5), 2.0)


def _supports_instructions(model: str) -> bool:
    """OpenAI's ``instructions`` field works only with gpt-4o TTS models.

    tts-1 and tts-1-hd don't accept it, and other compatible services may
    not either, so it is only sent to model names like gpt-4o-mini-tts.
    """
    name = model.lower()
    return "gpt-4o" in name and "tts" in name


class SarvamTTSProvider(TTSProvider):
    """Narrate via Sarvam AI Bulbul (`POST {base}/text-to-speech`).

    Verified against Sarvam docs: `api-subscription-key` header, BCP-47
    `kn-IN` language, `bulbul:v3` model, lowercase speaker names, base64
    audio in the JSON response. Two naming notes encoded below: the REST
    reference shows `language_code` while the official SDK sends
    `target_language_code` — the request tries the SDK name first and falls
    back once on a 422. Response audio is read defensively (`audios` list,
    else any base64-looking string that decodes to WAV bytes).

    Bulbul has no emotion parameter (v3 supports `pace` and `temperature`,
    not pitch or loudness), so with `use_emotion` the tag becomes a small
    pace change (see `_SARVAM_EMOTION_PACE`); `voice` selects the speaker
    (e.g. "anushka").
    """

    def __init__(
        self,
        api_key: str,
        voice: str = "anushka",
        model: str = "bulbul:v3",
        language_code: str = "kn-IN",
        base_url: str = "https://api.sarvam.ai",
        sampling_rate: int = 22050,
        max_chars_per_call: int = 2000,
        use_emotion: bool = True,
    ):
        self._use_emotion = use_emotion
        self._api_key = api_key
        self._voice = voice.lower()
        self._model = model
        self._language_code = language_code
        self._base_url = base_url.rstrip("/")
        self.sampling_rate = sampling_rate
        self._max_chars = max_chars_per_call

    def _convert(self, text: str, language_field: str, pace: float = 1.0) -> httpx.Response:
        body = {
            "text": text,
            language_field: self._language_code,
            "speaker": self._voice,
            "model": self._model,
            "speech_sample_rate": self.sampling_rate,
        }
        if pace != 1.0:
            body["pace"] = pace
        return httpx.post(
            f"{self._base_url}/text-to-speech",
            headers={"api-subscription-key": self._api_key},
            json=body,
            timeout=120,
        )

    @staticmethod
    def _extract_audio(payload: dict) -> bytes:
        if isinstance(payload.get("audios"), list) and payload["audios"]:
            return base64.b64decode(payload["audios"][0])
        for value in payload.values():
            if isinstance(value, str) and len(value) > 100 and _B64_RE.match(value):
                try:
                    raw = base64.b64decode(value)
                except Exception:
                    continue
                if raw[:4] == b"RIFF":
                    return raw
        raise RuntimeError(f"Sarvam TTS response held no decodable audio. Keys: {sorted(payload.keys())}")

    def _synthesize_chunk(self, text: str, pace: float = 1.0) -> np.ndarray:
        response = self._convert(text, "target_language_code", pace)
        if response.status_code == 422:
            response = self._convert(text, "language_code", pace)
        response.raise_for_status()
        audio, rate = _decode_audio_bytes(self._extract_audio(response.json()))
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.9
        return _resample(audio, rate, self.sampling_rate)

    def synthesize_paragraph(self, text: str, voice: str, emotion: str) -> np.ndarray:
        pace = _sarvam_pace(emotion) if self._use_emotion else 1.0
        chunks = _split_for_limit(text, self._max_chars)
        gap = np.zeros(int(0.25 * self.sampling_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                pieces.append(gap)
            pieces.append(self._synthesize_chunk(chunk, pace))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


class OpenAICompatibleTTSProvider(TTSProvider):
    """Narrate via any OpenAI-compatible `/audio/speech` endpoint.

    Standard schema (`{model, input, voice, response_format: "wav"}` →
    raw audio bytes). With `use_emotion`, gpt-4o TTS models also get an
    `instructions` line describing the paragraph's emotion; other models
    (tts-1, other services) don't accept that field, so for them the tag is
    ignored. `voice` passes through verbatim.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        sampling_rate: int = 24000,
        max_chars_per_call: int = 4000,
        use_emotion: bool = True,
    ):
        self._use_emotion = use_emotion
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._default_voice = voice
        self.sampling_rate = sampling_rate
        self._max_chars = max_chars_per_call

    def _synthesize_chunk(self, text: str, voice: str, emotion: str = "Narration") -> np.ndarray:
        body = {
            "model": self._model,
            "input": text,
            "voice": voice or self._default_voice,
            "response_format": "wav",
        }
        if self._use_emotion and _supports_instructions(self._model):
            phrase = EMOTION_PHRASES.get(emotion, EMOTION_PHRASES["Narration"])
            body["instructions"] = (
                f"Read this Kannada text aloud like an audiobook narrator, in {phrase}, "
                "at a moderate pace."
            )
        response = httpx.post(
            f"{self._base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        audio, rate = _decode_audio_bytes(response.content)
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.9
        return _resample(audio, rate, self.sampling_rate)

    def synthesize_paragraph(self, text: str, voice: str, emotion: str) -> np.ndarray:
        chunks = _split_for_limit(text, self._max_chars)
        gap = np.zeros(int(0.25 * self.sampling_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                pieces.append(gap)
            pieces.append(self._synthesize_chunk(chunk, voice, emotion))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
