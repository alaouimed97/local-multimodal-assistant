"""Unit tests for the pipeline routing, with Ollama and Whisper replaced by fakes."""

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from assistant import InferenceTracker, MultimodalAssistant, Settings, Transcription
from assistant.config import DEFAULT_AUDIO_PROMPT


class FakeOllama:
    def __init__(self, reply="A calm lake at sunset.", capabilities=("completion", "vision", "audio", "thinking"),
                 thinking=""):
        self.reply = reply
        self.thinking = thinking
        self._capabilities = list(capabilities)
        self.calls = []

    def show(self, model):
        return SimpleNamespace(capabilities=self._capabilities)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.thinking:
            yield SimpleNamespace(message=SimpleNamespace(content="", thinking=self.thinking), done=False)
        for word in self.reply.split(" "):
            yield SimpleNamespace(message=SimpleNamespace(content=word + " ", thinking=None), done=False)
        yield SimpleNamespace(
            message=SimpleNamespace(content="", thinking=None), done=True,
            load_duration=0, prompt_eval_count=12, prompt_eval_duration=50_000_000,
            eval_count=6, eval_duration=100_000_000,
        )


class FakeWhisper:
    def __init__(self, text="What is in this picture?"):
        self.text = text
        self.paths = []

    def transcribe(self, path):
        self.paths.append(path)
        return Transcription(self.text, "en", 0.99, 2.0, 0.25)


@pytest.fixture
def make_assistant(tmp_path):
    def factory(client=None, whisper=None):
        client = client or FakeOllama()
        whisper = whisper or FakeWhisper()
        settings = Settings(model="test-model", log_path=tmp_path / "log.jsonl")
        assistant = MultimodalAssistant(settings, client=client, transcriber=whisper,
                                        tracker=InferenceTracker(settings.log_path))
        return assistant, client, whisper
    return factory


def write_wav(path, rate=44_100, channels=2, seconds=0.5):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    frames = np.repeat(tone[:, None], channels, axis=1).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return str(path)


def user_message(client):
    return client.calls[-1]["messages"][-1]


def test_text_only(make_assistant):
    assistant, client, whisper = make_assistant()
    answer = assistant.ask("Hello there")
    msg = user_message(client)
    assert msg == {"role": "user", "content": "Hello there"}
    assert client.calls[-1]["messages"][0]["role"] == "system"
    assert answer.text == "A calm lake at sunset."
    assert whisper.paths == []


def test_image_is_normalised_to_jpeg_with_default_prompt(make_assistant, tmp_path):
    assistant, client, _ = make_assistant()
    path = tmp_path / "pic.webp"
    Image.new("RGBA", (3000, 1000), (255, 0, 0, 128)).save(path)
    assistant.ask(image_path=str(path))
    msg = user_message(client)
    assert msg["content"] == "Describe this image in detail."
    jpeg = msg["images"][0]
    assert jpeg[:2] == b"\xff\xd8"
    assert max(Image.open(io.BytesIO(jpeg)).size) == 1600


def test_whisper_pipeline_sends_transcript_as_text(make_assistant, tmp_path):
    assistant, client, whisper = make_assistant()
    audio = write_wav(tmp_path / "q.wav")
    events = list(assistant.stream(audio_path=audio, audio_pipeline="whisper"))
    assert events[0].kind == "transcript" and events[0].text == "What is in this picture?"
    msg = user_message(client)
    assert msg == {"role": "user", "content": "What is in this picture?"}
    record = events[-1].record
    assert record.modalities == ["audio"] and record.audio_pipeline == "whisper"
    assert record.stt_seconds == 0.25 and record.transcript == "What is in this picture?"


def test_typed_text_and_transcript_are_combined_with_image(make_assistant, tmp_path):
    assistant, client, _ = make_assistant()
    image = tmp_path / "pic.png"
    Image.new("RGB", (64, 64), "blue").save(image)
    assistant.ask("Be brief.", str(image), write_wav(tmp_path / "q.wav"))
    msg = user_message(client)
    assert msg["content"] == "Be brief.\n\nWhat is in this picture?"
    assert len(msg["images"]) == 1


def test_native_audio_is_resampled_to_16k_mono_wav(make_assistant, tmp_path):
    assistant, client, whisper = make_assistant()
    assistant.ask(audio_path=write_wav(tmp_path / "q.wav"), audio_pipeline="native")
    msg = user_message(client)
    assert whisper.paths == []
    with wave.open(io.BytesIO(msg["images"][0])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
    assert msg["content"] == DEFAULT_AUDIO_PROMPT


def test_native_audio_heard_line_is_reported_separately(make_assistant, tmp_path):
    client = FakeOllama(reply="Heard: What is this?\n\nIt is a lake at sunset.")
    assistant, _, _ = make_assistant(client=client)
    events = list(assistant.stream(audio_path=write_wav(tmp_path / "q.wav"), audio_pipeline="native"))
    assert [e.text for e in events if e.kind == "heard"] == ["What is this?"]
    streamed = "".join(e.text for e in events if e.kind == "content")
    assert streamed.strip() == "It is a lake at sunset."
    done = events[-1]
    assert done.text == "It is a lake at sunset."
    assert done.record.transcript == "What is this?"
    assert done.turn[0] == {"role": "user", "content": "What is this?"}


def test_native_audio_without_heard_line_streams_unchanged(make_assistant, tmp_path):
    assistant, _, _ = make_assistant(client=FakeOllama(reply="It is a lake at sunset."))
    answer = assistant.ask(audio_path=write_wav(tmp_path / "q.wav"), audio_pipeline="native")
    assert answer.heard is None and answer.text == "It is a lake at sunset."


@pytest.mark.parametrize("reply, expected", [
    ("Heard: What is this?\nA lake.", ("What is this?", "A lake.")),
    ('**Heard:** "What is this?"\n\nA lake.', ("What is this?", "A lake.")),
    ("Heard:\nWhat is this?\nA lake.", ("What is this?", "A lake.")),
    ("A lake, no preamble.", (None, "A lake, no preamble.")),
])
def test_split_heard(reply, expected):
    from assistant.core import split_heard
    assert split_heard(reply) == expected


def test_image_comes_before_native_audio(make_assistant, tmp_path):
    assistant, client, _ = make_assistant()
    image = tmp_path / "pic.png"
    Image.new("RGB", (64, 64), "green").save(image)
    assistant.ask(image_path=str(image), audio_path=write_wav(tmp_path / "q.wav"), audio_pipeline="native")
    first, second = user_message(client)["images"]
    assert first[:2] == b"\xff\xd8" and second[:4] == b"RIFF"


@pytest.mark.parametrize("capabilities, think, expected", [
    (("completion", "thinking"), False, {"think": False}),
    (("completion", "thinking"), True, {"think": True}),
    (("completion",), False, {"think": False}),
    (("completion",), True, {}),
])
def test_think_flag(make_assistant, capabilities, think, expected):
    assistant, client, _ = make_assistant(client=FakeOllama(capabilities=capabilities))
    assistant.ask("hi", think=think)
    sent = {k: v for k, v in client.calls[-1].items() if k == "think"}
    assert sent == expected


def test_thinking_is_streamed_separately(make_assistant):
    assistant, _, _ = make_assistant(client=FakeOllama(thinking="Let me look..."))
    answer = assistant.ask("hi", think=True)
    assert answer.thinking == "Let me look..."
    assert answer.text == "A calm lake at sunset."


def test_empty_input_is_rejected(make_assistant):
    assistant, client, _ = make_assistant()
    with pytest.raises(ValueError):
        assistant.ask("   ")
    assert client.calls == []


def test_silent_recording_without_other_input_is_rejected(make_assistant, tmp_path):
    assistant, client, _ = make_assistant(whisper=FakeWhisper(text=""))
    with pytest.raises(ValueError, match="No speech"):
        assistant.ask(audio_path=write_wav(tmp_path / "q.wav"))
    assert client.calls == []
    assert assistant.tracker.records()[-1]["error"].startswith("ValueError")


def test_metrics_are_tracked(make_assistant):
    assistant, _, _ = make_assistant()
    answer = assistant.ask("hi")
    assert answer.record.output_tokens == 6
    assert answer.record.tokens_per_second == pytest.approx(60.0)
    lines = assistant.tracker.path.read_text(encoding="utf-8").splitlines()
    logged = json.loads(lines[-1])
    assert logged["prompt_tokens"] == 12 and logged["modalities"] == ["text"]
    summary = assistant.tracker.summary()
    assert summary["requests"] == 1 and summary["by_modality"] == {"text": 1}


def test_history_is_trimmed_and_turn_returned(make_assistant):
    assistant, client, _ = make_assistant()
    history = []
    for i in range(6):
        history += assistant.ask(f"question {i}", history=history).turn
    sent = client.calls[-1]["messages"]
    # system + last `history_turns` (4) exchanges + the new question
    assert len(sent) == 1 + 2 * assistant.settings.history_turns + 1
    assert sent[1]["content"] == "question 1"
    assert history[-2:] == [{"role": "user", "content": "question 5"},
                            {"role": "assistant", "content": "A calm lake at sunset."}]
