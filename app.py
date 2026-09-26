"""Web UI for the local multimodal assistant (Gradio, bound to 127.0.0.1 only).

    python app.py            ->  http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import threading

# Privacy: no Gradio usage analytics / Hugging Face telemetry leave the machine.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr  # noqa: E402

from assistant import MultimodalAssistant  # noqa: E402
from assistant.config import ROOT  # noqa: E402

assistant = MultimodalAssistant()
settings = assistant.settings

AUDIO_PIPELINES = {
    "Whisper → Gemma 4 (recommended)": "whisper",
    "Gemma 4 native audio (experimental)": "native",
}

CSS = """
.app-header {padding: 4px 2px 10px}
.app-header h1 {margin: 0; font-size: 1.55rem}
.app-header p {margin: 4px 0 8px; opacity: .75}
.pill {display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
       font-size: .8rem; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary)}
.metrics table {font-size: .85rem}
"""


def header_html() -> str:
    try:
        version = assistant.server_version()
        server = f"Ollama v{version} @ {settings.ollama_host}"
    except Exception:
        server = f"Ollama not reachable at {settings.ollama_host}"
    pills = [
        f"🧠 {settings.model}",
        f"🦙 {server}",
        f"🎙️ Whisper {settings.whisper_model} ({settings.whisper_device}, {settings.whisper_compute_type})",
        "🔒 100% local · no cloud calls",
    ]
    return (
        "<div class='app-header'><h1>Local Multimodal Assistant</h1>"
        "<p>Ask with text, images or your voice. Everything runs on this computer.</p>"
        + "".join(f"<span class='pill'>{p}</span>" for p in pills)
        + "</div>"
    )


def metrics_markdown(record) -> str:
    rows = [
        ("Modalities", " + ".join(record.modalities)),
        ("Audio pipeline", {"whisper": "Whisper STT → Gemma 4", "native": "Gemma 4 native audio"}.get(
            record.audio_pipeline, "—")),
        ("Whisper time", f"{record.stt_seconds:.2f} s" if record.stt_seconds is not None else "—"),
        ("Prompt tokens", f"{record.prompt_tokens} (in {record.prompt_eval_seconds:.2f} s)"),
        ("Output tokens", str(record.output_tokens)),
        ("Generation speed", f"{record.tokens_per_second:.1f} tok/s"),
        ("Model load", f"{record.load_seconds:.2f} s"),
        ("Total latency", f"{record.wall_seconds:.2f} s"),
    ]
    return "**Last inference**\n\n| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def respond(text, image, audio, pipeline_label, think, chat, history):
    text = (text or "").strip()
    if not (text or image or audio):
        gr.Warning("Type a question, add an image, or record/upload a voice message first.")
        yield chat, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    pipeline = AUDIO_PIPELINES[pipeline_label]
    user_content: list = []
    if image:
        user_content.append({"path": image})
    if audio:
        user_content.append({"path": audio})
    if text:
        user_content.append(text)
    chat = chat + [{"role": "user", "content": user_content}]
    if audio and pipeline == "whisper":
        status = "🎙️ Transcribing voice with Whisper…"
    elif audio:
        status = "🎧 Gemma 4 is listening to the raw audio…"
    else:
        status = "…"
    chat.append({"role": "assistant", "content": status})
    yield chat, history, gr.skip(), "", None, None

    thinking_msg = None
    answer = ""
    try:
        for event in assistant.stream(text, image, audio, history=history,
                                      audio_pipeline=pipeline, think=think):
            if event.kind == "transcript":
                t = event.transcription
                heard = t.text or "(no speech detected)"
                user_content.append(
                    f"🎙️ **Voice → text** (Whisper · {t.language} · {t.elapsed_seconds:.1f} s): “{heard}”")
                chat[-2] = {"role": "user", "content": user_content}
                chat[-1] = {"role": "assistant", "content": "…"}
            elif event.kind == "heard":
                user_content.append(f"🎧 **Gemma 4 heard** (native audio, no Whisper): “{event.text}”")
                chat[-2] = {"role": "user", "content": user_content}
            elif event.kind == "thinking":
                if thinking_msg is None:
                    thinking_msg = {"role": "assistant", "content": "",
                                    "metadata": {"title": "🧠 Thinking", "status": "pending"}}
                    chat.insert(len(chat) - 1, thinking_msg)
                thinking_msg["content"] += event.text
            elif event.kind == "content":
                if thinking_msg is not None:
                    thinking_msg["metadata"]["status"] = "done"
                answer += event.text
                chat[-1] = {"role": "assistant", "content": answer}
            elif event.kind == "done":
                record = event.record
                if thinking_msg is not None:
                    thinking_msg["metadata"]["status"] = "done"
                chat[-1] = {"role": "assistant",
                            "content": f"{event.text}\n\n<sub>⏱ {record.summary_line()}</sub>"}
                history = history + event.turn
                yield chat, history, metrics_markdown(record), gr.skip(), gr.skip(), gr.skip()
                return
            yield chat, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    except Exception as exc:  # surface the problem in the chat instead of a silent failure
        chat[-1] = {"role": "assistant", "content": f"⚠️ **Error:** {exc}"}
        yield chat, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()


def maybe_auto_send(auto, text, image, audio, pipeline_label, think, chat, history):
    if not auto:
        yield chat, history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    yield from respond(text, image, audio, pipeline_label, think, chat, history)


def tracking_view():
    try:
        version = assistant.server_version()
        loaded = assistant.loaded_models()
        installed = assistant.installed_models()
        lines = [
            "### Local deployment",
            f"- **Ollama server:** v{version} at `{settings.ollama_host}` (loopback only)",
            f"- **Assistant model:** `{settings.model}` — capabilities: {', '.join(assistant.capabilities)}",
            f"- **Speech-to-text:** faster-whisper `{settings.whisper_model}` on {settings.whisper_device} "
            f"({settings.whisper_compute_type})",
            "",
            "| Loaded model | Size in memory | On GPU | Context | Unloads at |",
            "|---|---|---|---|---|",
        ]
        for m in loaded:
            gpu_share = (m.size_vram / m.size * 100) if m.size else 0
            until = m.expires_at.strftime("%H:%M:%S") if m.expires_at else "—"
            lines.append(f"| `{m.model}` | {m.size / 1e9:.2f} GB | {gpu_share:.0f}% | "
                         f"{getattr(m, 'context_length', None) or '—'} | {until} |")
        if not loaded:
            lines.append("| *(no model loaded — it loads on the first request)* | | | | |")
        lines.append("\n**Installed models:** " + ", ".join(
            f"`{m.model}` ({m.size / 1e9:.1f} GB)" for m in installed))
        status = "\n".join(lines)
    except Exception as exc:
        status = f"### Local deployment\n⚠️ Ollama is not reachable: {exc}"

    s = assistant.tracker.summary()
    by_modality = ", ".join(f"{k}: {v}" for k, v in s["by_modality"].items()) or "—"
    summary = (
        "### Inference summary\n"
        f"- **Requests logged:** {s['requests']} ({by_modality})\n"
        f"- **Average generation speed:** {s['avg_tokens_per_second']:.1f} tok/s\n"
        f"- **Average end-to-end latency:** {s['avg_latency_seconds']:.2f} s\n"
        f"- **Total output tokens:** {s['total_output_tokens']}\n"
        f"- **Log file:** `{assistant.tracker.path}`"
    )
    rows = [
        [
            r["timestamp"].replace("T", " "),
            " + ".join(r["modalities"]),
            r.get("audio_pipeline") or "",
            (r.get("transcript") or r["prompt"])[:80],
            r["prompt_tokens"],
            r["output_tokens"],
            round(r["tokens_per_second"], 1),
            round(r["stt_seconds"], 2) if r.get("stt_seconds") is not None else "",
            round(r["wall_seconds"], 2),
            r.get("error") or "ok",
        ]
        for r in reversed(assistant.tracker.records()[-50:])
    ]
    return status, summary, rows


LOG_HEADERS = ["Time", "Input", "Audio via", "Prompt / what was heard", "In tok",
               "Out tok", "tok/s", "STT s", "Total s", "Status"]
LOG_WIDTHS = ["13%", "10%", "8%", "33%", "6%", "6%", "6%", "6%", "6%", "6%"]

with gr.Blocks(title="Local Multimodal Assistant") as demo:
    gr.HTML(header_html())
    history = gr.State([])

    with gr.Tabs():
        with gr.Tab("Assistant"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        elem_id="chat",
                        height=620,
                        buttons=["copy"],
                        placeholder="**Ask anything** — type below, drop an image on the right, "
                                    "or record a voice question.",
                    )
                    with gr.Row():
                        # lines=1 keeps Enter = send; the box still grows up to max_lines.
                        text_in = gr.Textbox(
                            show_label=False, scale=5, lines=1, max_lines=6,
                            placeholder="Type your question… (Enter to send)",
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
                    clear_btn = gr.Button("🗑️ New conversation", size="sm")
                with gr.Column(scale=2):
                    image_in = gr.Image(
                        label="🖼️ Image (optional)", type="filepath",
                        sources=["upload", "clipboard", "webcam"], height=250, elem_id="image-input",
                    )
                    audio_in = gr.Audio(
                        label="🎙️ Voice / audio question (record or upload)",
                        sources=["microphone", "upload"], type="filepath", format="wav", elem_id="audio-input",
                    )
                    with gr.Accordion("Settings", open=True):
                        pipeline_in = gr.Radio(
                            list(AUDIO_PIPELINES), value=next(iter(AUDIO_PIPELINES)), label="Audio pipeline",
                        )
                        with gr.Row():
                            think_in = gr.Checkbox(label="Thinking mode (slower, more reasoning)", value=False)
                            auto_send_in = gr.Checkbox(label="Send when recording stops", value=True)
                    metrics_md = gr.Markdown("**Last inference**\n\n_No request yet._", elem_classes="metrics")
            gr.Examples(
                examples=[
                    ["What are the main benefits of running AI models locally?", None, None],
                    ["", str(ROOT / "samples" / "sunset_lake.jpg"), str(ROOT / "samples" / "voice_describe_image.wav")],
                    ["", None, str(ROOT / "samples" / "voice_question.wav")],
                ],
                inputs=[text_in, image_in, audio_in],
                label="Try an example",
            )

        with gr.Tab("Inference tracking") as tracking_tab:
            status_md = gr.Markdown()
            summary_md = gr.Markdown()
            log_df = gr.Dataframe(headers=LOG_HEADERS, column_widths=LOG_WIDTHS,
                                  label="Request log (newest first)", interactive=False, wrap=True)
            refresh_btn = gr.Button("🔄 Refresh")

    chat_inputs = [text_in, image_in, audio_in, pipeline_in, think_in, chatbot, history]
    chat_outputs = [chatbot, history, metrics_md, text_in, image_in, audio_in]
    send_btn.click(respond, chat_inputs, chat_outputs)
    text_in.submit(respond, chat_inputs, chat_outputs)
    audio_in.stop_recording(maybe_auto_send, [auto_send_in, *chat_inputs], chat_outputs)
    clear_btn.click(lambda: ([], [], "**Last inference**\n\n_No request yet._", "", None, None),
                    None, chat_outputs)

    tracking_outputs = [status_md, summary_md, log_df]
    tracking_tab.select(tracking_view, None, tracking_outputs)
    refresh_btn.click(tracking_view, None, tracking_outputs)
    demo.load(tracking_view, None, tracking_outputs)


if __name__ == "__main__":
    # Load Whisper in the background so the first voice question doesn't pay for it.
    threading.Thread(target=assistant.transcriber.warm_up, daemon=True).start()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
        css=CSS,
        allowed_paths=[str(ROOT / "samples")],
    )
