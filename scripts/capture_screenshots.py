"""Drive the running web UI in Microsoft Edge (Playwright) and save screenshots of real
multimodal interactions to screenshots/. Nothing is mocked: files are uploaded through
the UI and the answers come from the local Gemma 4 model.

    python app.py                              # terminal 1
    python scripts/capture_screenshots.py      # terminal 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# Show the whole conversation in one image instead of the scrollable 620 px chat panel.
EXPAND_CHAT = """
#chat { height: auto !important; max-height: none !important; }
#chat .bubble-wrap { height: auto !important; max-height: none !important; overflow: visible !important; }
"""
# Every finished answer ends with a <sub>⏱ … tok/s</sub> metrics line.
ANSWER_DONE = "() => [...document.querySelectorAll('#chat sub')].some(s => s.textContent.includes('tok/s'))"
WHISPER = "Whisper → Gemma 4 (recommended)"
NATIVE = "Gemma 4 native audio (experimental)"


class UI:
    def __init__(self, page: Page, url: str, out: Path):
        self.page, self.url, self.out = page, url, out

    def fresh(self, pipeline: str = WHISPER) -> "UI":
        self.page.goto(self.url)
        self.page.wait_for_selector("#send-btn")
        self.page.add_style_tag(content=EXPAND_CHAT)
        self.page.get_by_label(pipeline).check()
        return self

    def image(self, name: str) -> "UI":
        self.page.locator("#image-input input[type=file]").set_input_files(SAMPLES / name)
        self.page.wait_for_selector("#image-input img")
        return self

    def audio(self, name: str) -> "UI":
        self.page.locator("#audio-input button[aria-label='Upload file']").click()
        with self.page.expect_response(lambda r: "/upload" in r.url and r.request.method == "POST"):
            self.page.locator("#audio-input input[type=file]").set_input_files(SAMPLES / name)
        self.page.wait_for_timeout(1000)  # let the waveform player render
        return self

    def send(self, text: str = "") -> "UI":
        if text:
            self.page.get_by_placeholder("Type your question").fill(text)
        self.page.locator("#send-btn").click()
        self.page.wait_for_function(ANSWER_DONE, timeout=300_000)
        self.page.wait_for_timeout(700)
        return self

    def shot(self, name: str) -> None:
        path = self.out / name
        self.page.screenshot(path=path, full_page=True)
        print(f"saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--out", type=Path, default=ROOT / "screenshots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1.5,
                                color_scheme="light")
        ui = UI(page, args.url, args.out)

        ui.fresh().send("Explain in three short bullet points why running an AI assistant "
                        "locally is good for privacy.").shot("01_text_query.png")
        ui.fresh().image("sunset_lake.jpg").send(
            "What time of day is shown here, and which visual details support that?").shot("02_image_query.png")
        ui.fresh().image("sunset_lake.jpg").audio("voice_describe_image.wav").send().shot(
            "03_voice_describes_image_whisper.png")
        ui.fresh().audio("voice_question.wav").send().shot("04_voice_question_whisper.png")
        ui.fresh(NATIVE).audio("voice_question.wav").send().shot("05_voice_question_native_audio.png")
        ui.fresh().image("sunset_lake.jpg").audio("voice_question_fr.wav").send().shot(
            "06_french_voice_about_image.png")

        ui.fresh()
        page.get_by_role("tab", name="Inference tracking").click()
        page.wait_for_selector("text=Local deployment")
        page.wait_for_timeout(1500)
        ui.shot("07_inference_tracking.png")
        browser.close()


if __name__ == "__main__":
    main()
