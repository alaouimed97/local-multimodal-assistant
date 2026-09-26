"""Runtime settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a private multimodal assistant running entirely offline on the user's "
    "computer through Ollama. Users can send you text, images and voice messages. "
    "Voice messages are either transcribed locally by Whisper (shown to you as a "
    "transcript) or passed to you as raw audio. Answer clearly and concisely, and "
    "when describing an image, only mention details that are actually visible."
)

DEFAULT_IMAGE_PROMPT = "Describe this image in detail."
# Asking for a "Heard:" line first made native audio answers more reliable in our trials
# (6/6 clean answers vs 5/6 for simpler prompts) and shows the user what the model understood.
DEFAULT_AUDIO_PROMPT = (
    "The attached audio is the user's spoken message. On the first line write "
    "'Heard: ' followed by exactly what they said, then answer it."
)


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: os.getenv("ASSISTANT_MODEL", "gemma4-assistant"))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("ASSISTANT_NUM_CTX", "8192")))
    # Number of previous user/assistant turns re-sent to the model as context.
    history_turns: int = field(default_factory=lambda: int(os.getenv("ASSISTANT_HISTORY_TURNS", "4")))
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    log_path: Path = field(default_factory=lambda: Path(
        os.getenv("ASSISTANT_LOG", str(ROOT / "logs" / "inference_log.jsonl"))))
