"""Local speech-to-text with faster-whisper (open-source Whisper on CTranslate2)."""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000

# Windows without Developer Mode cannot create the symlinks the HF cache prefers; it
# falls back to copies, which is fine, so the warning is only noise.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    language_probability: float
    audio_seconds: float
    elapsed_seconds: float


class WhisperTranscriber:
    """Loads the Whisper model lazily so the UI starts instantly."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                kwargs = dict(device=self.device, compute_type=self.compute_type)
                try:
                    # Offline first: once the weights are cached no request leaves the machine.
                    self._model = WhisperModel(self.model_size, local_files_only=True, **kwargs)
                except Exception:
                    # First run only: fetch the open Whisper weights from Hugging Face.
                    self._model = WhisperModel(self.model_size, **kwargs)
            return self._model

    def warm_up(self) -> None:
        """Load the model ahead of the first voice question (safe to call from a thread)."""
        self._get_model()

    def transcribe(self, audio_path: str) -> Transcription:
        model = self._get_model()
        start = time.perf_counter()
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcription(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            audio_seconds=info.duration,
            elapsed_seconds=time.perf_counter() - start,
        )


def to_wav_16k_mono(audio_path: str) -> bytes:
    """Decode any audio/video file (wav, mp3, m4a, webm, ...) to 16 kHz mono PCM WAV bytes,
    the format Gemma 4's native audio encoder expects."""
    from faster_whisper import decode_audio

    samples = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()
