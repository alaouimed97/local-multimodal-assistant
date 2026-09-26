"""Terminal client for the local multimodal assistant.

One-shot:
    python cli.py "What is Ollama?"
    python cli.py --image samples/sunset_lake.jpg "What time of day is it?"
    python cli.py --image samples/sunset_lake.jpg --audio samples/voice_describe_image.wav
    python cli.py --audio samples/voice_question.wav --native-audio

Interactive (type text; `/image PATH` attaches an image to the next question,
`/audio PATH` asks a recorded voice question, `/reset`, `/quit`):
    python cli.py -i
"""

from __future__ import annotations

import argparse
import sys

from assistant import MultimodalAssistant


def run_turn(assistant, history, text="", image=None, audio=None, native=False, think=False):
    in_thinking = False
    for event in assistant.stream(text, image, audio, history=history,
                                  audio_pipeline="native" if native else "whisper", think=think):
        if event.kind == "transcript":
            t = event.transcription
            print(f"[whisper {t.language} {t.elapsed_seconds:.1f}s] you said: {t.text!r}")
        elif event.kind == "heard":
            print(f"[gemma 4 native audio] heard: {event.text!r}")
        elif event.kind == "thinking":
            if not in_thinking:
                print("[thinking] ", end="")
                in_thinking = True
            print(event.text, end="", flush=True)
        elif event.kind == "content":
            if in_thinking:
                print("\n")
                in_thinking = False
            print(event.text, end="", flush=True)
        elif event.kind == "done":
            print(f"\n[{event.record.summary_line()}]")
            history.extend(event.turn)


def interactive(assistant, native, think):
    history: list[dict] = []
    image = None
    print(f"Local assistant ({assistant.settings.model}). /image PATH, /audio PATH, /reset, /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/reset":
            history.clear()
            image = None
            print("(conversation cleared)")
            continue
        try:
            if line.startswith("/image "):
                image = line[len("/image "):].strip().strip('"')
                print(f"(image attached to your next question: {image})")
                continue
            if line.startswith("/audio "):
                run_turn(assistant, history, audio=line[len("/audio "):].strip().strip('"'),
                         image=image, native=native, think=think)
            else:
                run_turn(assistant, history, text=line, image=image, native=native, think=think)
            image = None
        except Exception as exc:
            print(f"error: {exc}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Local multimodal assistant (Gemma 4 on Ollama + Whisper).")
    parser.add_argument("text", nargs="?", default="", help="text question")
    parser.add_argument("--image", help="path to an image file")
    parser.add_argument("--audio", help="path to a voice/audio recording (wav, mp3, m4a, ...)")
    parser.add_argument("--native-audio", action="store_true",
                        help="send audio straight to Gemma 4 instead of transcribing it with Whisper")
    parser.add_argument("--think", action="store_true", help="enable Gemma 4 thinking mode")
    parser.add_argument("-i", "--interactive", action="store_true", help="start an interactive session")
    args = parser.parse_args()

    assistant = MultimodalAssistant()
    if args.interactive:
        interactive(assistant, args.native_audio, args.think)
    elif args.text or args.image or args.audio:
        run_turn(assistant, [], args.text, args.image, args.audio, args.native_audio, args.think)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
