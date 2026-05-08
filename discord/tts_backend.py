"""TTS backend abstraction (A3).

Backends yield audio chunk-by-chunk so the call site can either batch
(for `/discord/speak` file output) or stream (future `/discord/speak_voice`
upgrade — see plan A3 for the QueuedPCMAudio direction).

Backends:
  - PiperBackend     — current default; library-bound piper.voice.PiperVoice
  - KokoroOnnxBackend — kokoro-onnx package (ONNX runtime, ~310MB model)
  - KokoroTorchBackend — kokoro package (PyTorch path)

Selection: cfg.tts.backend ∈ {"piper", "kokoro_onnx", "kokoro_torch"}.
Voice / model paths via env vars (KOKORO_VOICE, KOKORO_MODEL, PIPER_MODEL).
"""

from __future__ import annotations

import os
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSBackend(Protocol):
    """Yield (int16 mono samples, sample_rate) chunks per natural unit
    (typically a sentence). Caller is responsible for resampling to Discord's
    48 kHz stereo via the shared `_to_discord_pcm` helper in main.py.
    """

    def synthesize(self, text: str) -> Iterator[tuple[np.ndarray, int]]: ...


# ── Piper (current default) ─────────────────────────────────────────────────

class PiperBackend:
    """Library-bound singleton. Mirrors the original main.py inline path."""

    def __init__(self) -> None:
        from piper.voice import PiperVoice
        model = os.environ.get("PIPER_MODEL", "/models/en_US-ryan-low.onnx")
        self._voice = PiperVoice.load(model)

    def synthesize(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        # piper's .synthesize() is already a generator yielding per-sentence
        # AudioChunk(audio_float_array, sample_rate). The original main.py
        # collapsed the stream via list() — keep the streaming shape here so
        # downstream voice playback can later push chunks live.
        for chunk in self._voice.synthesize(text):
            audio = np.clip(chunk.audio_float_array, -1.0, 1.0)
            pcm16 = (audio * 32767).astype(np.int16)
            yield pcm16, chunk.sample_rate


# ── Kokoro ONNX ─────────────────────────────────────────────────────────────

class KokoroOnnxBackend:
    """kokoro-onnx package — matches Piper's ONNX inference shape."""

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro
        model_path = os.environ.get("KOKORO_MODEL", "/models/kokoro-v1.0.onnx")
        voices_path = os.environ.get("KOKORO_VOICES", "/models/voices-v1.0.bin")
        self._kokoro = Kokoro(model_path, voices_path)
        self._voice = os.environ.get("KOKORO_VOICE", "af_bella")

    def synthesize(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        # kokoro-onnx returns (samples_float32, sample_rate). For now we emit
        # one chunk per call; a future PR can split text into sentences and
        # yield sentence-by-sentence to match Piper's streaming granularity.
        samples, sample_rate = self._kokoro.create(text, voice=self._voice)
        audio = np.clip(samples, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16)
        yield pcm16, int(sample_rate)


# ── Kokoro PyTorch ──────────────────────────────────────────────────────────

class KokoroTorchBackend:
    """kokoro PyPI package — PyTorch path, heavier runtime, kept available
    for quality A/B vs the ONNX path."""

    def __init__(self) -> None:
        from kokoro import KPipeline
        # 'a' = American English; matches af_* / am_* voice prefixes.
        # User can override via KOKORO_LANG.
        lang_code = os.environ.get("KOKORO_LANG", "a")
        self._pipeline = KPipeline(lang_code=lang_code)
        self._voice = os.environ.get("KOKORO_VOICE", "af_bella")

    def synthesize(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        # KPipeline yields per-sentence GeneratorOutput(audio: tensor) at 24 kHz.
        for out in self._pipeline(text, voice=self._voice):
            audio_np = out.audio.numpy() if hasattr(out.audio, "numpy") else np.asarray(out.audio)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm16 = (audio_np * 32767).astype(np.int16)
            yield pcm16, 24000


# ── Factory ─────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type] = {
    "piper":        PiperBackend,
    "kokoro_onnx":  KokoroOnnxBackend,
    "kokoro_torch": KokoroTorchBackend,
}

_cached: dict[str, TTSBackend] = {}


def get_backend(name: str) -> TTSBackend:
    """Return a singleton backend by config-key name. Loads lazily so the
    container only pays init cost for the backend actually used."""
    name = (name or "piper").lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown TTS backend {name!r}. Valid: {sorted(_BACKENDS)}"
        )
    if name not in _cached:
        _cached[name] = _BACKENDS[name]()
    return _cached[name]
