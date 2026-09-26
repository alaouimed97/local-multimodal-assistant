"""End-to-end checks against the real local stack (Ollama + Gemma 4 + Whisper).

Skipped automatically when the Ollama server is not running.
    pytest -m integration
"""

from __future__ import annotations

import httpx
import pytest

from assistant import MultimodalAssistant, Settings
from assistant.config import ROOT

SAMPLES = ROOT / "samples"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def assistant(tmp_path_factory):
    settings = Settings(log_path=tmp_path_factory.mktemp("logs") / "log.jsonl")
    try:
        httpx.get(f"{settings.ollama_host}/api/version", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama server is not running")
    return MultimodalAssistant(settings)


def test_model_is_multimodal(assistant):
    assert {"vision", "audio"} <= set(assistant.capabilities)


def test_text(assistant):
    answer = assistant.ask("Reply with exactly one word: what is the capital of France?")
    assert "paris" in answer.text.lower()


def test_image(assistant):
    answer = assistant.ask("In one sentence, what is in this image?", str(SAMPLES / "sunset_lake.jpg"))
    assert any(word in answer.text.lower() for word in ("sun", "lake", "water", "snow"))
    assert answer.record.prompt_tokens > 200  # the image really reached the model


def test_voice_about_image_with_whisper(assistant):
    answer = assistant.ask(image_path=str(SAMPLES / "sunset_lake.jpg"),
                           audio_path=str(SAMPLES / "voice_describe_image.wav"))
    assert "describe this image" in answer.transcription.text.lower()
    assert any(word in answer.text.lower() for word in ("sun", "lake", "water", "snow"))


def test_native_audio(assistant):
    answer = assistant.ask(audio_path=str(SAMPLES / "voice_question.wav"), audio_pipeline="native")
    understood = (answer.heard or answer.text).lower()
    assert "local" in understood and "cloud" in understood
    assert answer.text and not answer.text.lower().startswith("heard")
