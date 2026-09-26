"""Private, local multimodal assistant: Gemma 4 on Ollama + Whisper speech-to-text."""

from .config import Settings
from .core import Answer, MultimodalAssistant, StreamEvent
from .speech import Transcription, WhisperTranscriber
from .tracking import InferenceRecord, InferenceTracker

__all__ = [
    "Answer",
    "InferenceRecord",
    "InferenceTracker",
    "MultimodalAssistant",
    "Settings",
    "StreamEvent",
    "Transcription",
    "WhisperTranscriber",
]
