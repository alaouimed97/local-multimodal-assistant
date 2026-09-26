"""Multimodal pipeline: text, image and voice in -> Gemma 4 (via Ollama) out.

Voice input takes one of two routes:
  * "whisper": faster-whisper transcribes locally, the transcript is sent as text.
  * "native":  the audio is resampled to 16 kHz mono WAV and given to Gemma 4's
               own audio encoder (Ollama accepts it alongside images).
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator, Literal

import httpx
import ollama
from PIL import Image, ImageOps

from .config import DEFAULT_AUDIO_PROMPT, DEFAULT_IMAGE_PROMPT, SYSTEM_PROMPT, Settings
from .speech import Transcription, WhisperTranscriber, to_wav_16k_mono
from .tracking import InferenceRecord, InferenceTracker

AudioPipeline = Literal["whisper", "native"]
MAX_IMAGE_SIDE = 1600
# First line of a native-audio answer, e.g. "Heard: What is ...?" or "**Heard:** ...".
HEARD_LINE = re.compile(r"^[\s*_#>]*heard[\s*_]*:[\s*_]*(?P<heard>.*?)[\s*_]*$", re.IGNORECASE)


@dataclass
class StreamEvent:
    # transcript: Whisper result · heard: what Gemma 4 understood from raw audio
    kind: Literal["transcript", "heard", "thinking", "content", "done"]
    text: str = ""
    transcription: Transcription | None = None
    record: InferenceRecord | None = None
    # On "done": the user and assistant messages to append to the conversation history.
    turn: list[dict] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    thinking: str
    transcription: Transcription | None
    record: InferenceRecord
    turn: list[dict]
    heard: str | None = None


def split_heard(text: str) -> tuple[str | None, str]:
    """Split a native-audio reply into (what the model heard, the actual answer)."""
    # Only left-strip the answer: while streaming, its trailing space separates it from the next chunk.
    lines = text.lstrip().split("\n")
    match = HEARD_LINE.match(lines[0])
    if not match:
        return None, text.lstrip()
    heard, rest = match["heard"], lines[1:]
    if not heard and rest:  # "Heard:" alone on its line, the words on the next one
        heard, rest = rest[0], rest[1:]
    return heard.strip().strip('"“”').strip(), "\n".join(rest).lstrip()


def heard_line_complete(buffer: str) -> bool:
    """True once enough of the reply has streamed to tell whether it opens with a Heard: line."""
    stripped = buffer.lstrip()
    if len(stripped) >= 10 and not re.match(r"[\s*_#>]*heard", stripped, re.IGNORECASE):
        return True
    lines = stripped.split("\n")
    if len(lines) < 2:
        return False
    match = HEARD_LINE.match(lines[0])
    return len(lines) >= 3 if match and not match["heard"] else True


def prepare_image(path: str) -> bytes:
    """Normalise any image (webp, png with alpha, huge photos, rotated phone shots)
    to an RGB JPEG the vision encoder can read."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


class MultimodalAssistant:
    def __init__(
        self,
        settings: Settings | None = None,
        client: ollama.Client | None = None,
        transcriber: WhisperTranscriber | None = None,
        tracker: InferenceTracker | None = None,
    ):
        self.settings = settings or Settings()
        self.client = client or ollama.Client(host=self.settings.ollama_host)
        self.transcriber = transcriber or WhisperTranscriber(
            self.settings.whisper_model, self.settings.whisper_device, self.settings.whisper_compute_type
        )
        self.tracker = tracker or InferenceTracker(self.settings.log_path)

    # ------------------------------------------------------------------ status
    @cached_property
    def capabilities(self) -> list[str]:
        return list(self.client.show(self.settings.model).capabilities or [])

    def server_version(self) -> str:
        response = httpx.get(f"{self.settings.ollama_host}/api/version", timeout=5)
        response.raise_for_status()
        return response.json()["version"]

    def loaded_models(self) -> list:
        return list(self.client.ps().models)

    def installed_models(self) -> list:
        return list(self.client.list().models)

    # ---------------------------------------------------------------- pipeline
    def stream(
        self,
        text: str = "",
        image_path: str | None = None,
        audio_path: str | None = None,
        *,
        history: list[dict] | None = None,
        audio_pipeline: AudioPipeline = "whisper",
        think: bool = False,
    ) -> Iterator[StreamEvent]:
        text = (text or "").strip()
        if not (text or image_path or audio_path):
            raise ValueError("Provide a text question, an image, or a voice/audio recording.")

        started = time.perf_counter()
        modalities = [name for name, value in (("text", text), ("image", image_path), ("audio", audio_path)) if value]
        record = InferenceRecord(
            model=self.settings.model,
            modalities=modalities,
            prompt="",
            audio_pipeline=audio_pipeline if audio_path else None,
            thinking=think,
        )
        try:
            prompt_parts = [text] if text else []
            images: list[bytes] = []
            transcription = None

            if audio_path and audio_pipeline == "whisper":
                transcription = self.transcriber.transcribe(audio_path)
                record.transcript = transcription.text
                record.stt_seconds = transcription.elapsed_seconds
                yield StreamEvent("transcript", transcription.text, transcription=transcription)
                if transcription.text:
                    prompt_parts.append(transcription.text)
                elif not (text or image_path):
                    raise ValueError("No speech was detected in the recording.")
            elif audio_path:
                images.append(to_wav_16k_mono(audio_path))
                if not text:
                    prompt_parts.append(DEFAULT_AUDIO_PROMPT)

            if image_path:
                images.insert(0, prepare_image(image_path))
                if not prompt_parts:
                    prompt_parts.append(DEFAULT_IMAGE_PROMPT)

            prompt = "\n\n".join(prompt_parts)
            record.prompt = prompt
            user_message = {"role": "user", "content": prompt}
            if images:
                user_message["images"] = images

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += (history or [])[-2 * self.settings.history_turns:]
            messages.append(user_message)

            # Gemma 4 reasons by default; `think=False` switches that off. Models without the
            # thinking capability reject `think=True`, so it is only sent when supported.
            extra = {"think": think} if "thinking" in self.capabilities or not think else {}
            content, thinking, final = [], [], None
            # With the default native-audio prompt the reply opens with "Heard: ...";
            # hold that line back and report it separately from the answer.
            holding = bool(audio_path and audio_pipeline == "native" and not text)
            held, heard = "", None
            for chunk in self.client.chat(
                model=self.settings.model,
                messages=messages,
                stream=True,
                options={"num_ctx": self.settings.num_ctx},
                **extra,
            ):
                if chunk.message.thinking:
                    thinking.append(chunk.message.thinking)
                    yield StreamEvent("thinking", chunk.message.thinking)
                if chunk.message.content:
                    content.append(chunk.message.content)
                    if not holding:
                        yield StreamEvent("content", chunk.message.content)
                    else:
                        held += chunk.message.content
                        if heard_line_complete(held) or len(held) > 500:
                            holding = False
                            heard, rest = split_heard(held)
                            yield from self._heard_events(heard, rest)
                if chunk.done:
                    final = chunk
            if holding and held:
                heard, rest = split_heard(held)
                yield from self._heard_events(heard, rest)

            answer = "".join(content).strip()
            if heard is not None:
                answer = split_heard(answer)[1]
                record.transcript = heard
            if final is not None:
                record.add_ollama_metrics(final)
            record.wall_seconds = time.perf_counter() - started
            record.response_preview = answer[:300]
            self.tracker.log(record)

            # Keep images (not raw audio) in the history so follow-up questions can refer to them.
            history_user = {"role": "user", "content": heard or prompt}
            if image_path:
                history_user["images"] = images[:1]
            yield StreamEvent(
                "done", answer, transcription=transcription, record=record,
                turn=[history_user, {"role": "assistant", "content": answer}],
            )
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.wall_seconds = time.perf_counter() - started
            self.tracker.log(record)
            raise

    @staticmethod
    def _heard_events(heard: str | None, rest: str) -> Iterator[StreamEvent]:
        if heard:
            yield StreamEvent("heard", heard)
        if rest:
            yield StreamEvent("content", rest)

    def ask(self, text: str = "", image_path: str | None = None, audio_path: str | None = None, **kwargs) -> Answer:
        """Blocking convenience wrapper around :meth:`stream`."""
        thinking, transcription, heard = [], None, None
        for event in self.stream(text, image_path, audio_path, **kwargs):
            if event.kind == "thinking":
                thinking.append(event.text)
            elif event.kind == "transcript":
                transcription = event.transcription
            elif event.kind == "heard":
                heard = event.text
            elif event.kind == "done":
                return Answer(event.text, "".join(thinking), transcription, event.record, event.turn, heard)
        raise RuntimeError("The model stream ended without a final response.")
