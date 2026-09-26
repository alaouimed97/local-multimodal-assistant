"""Inference tracking: every request is appended to a JSONL log with Ollama's
timing counters, so latency and throughput can be audited per modality."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

NS = 1e9


@dataclass
class InferenceRecord:
    model: str
    modalities: list[str]
    prompt: str
    audio_pipeline: str | None = None
    transcript: str | None = None
    thinking: bool = False
    stt_seconds: float | None = None
    wall_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_tokens: int = 0
    prompt_eval_seconds: float = 0.0
    output_tokens: int = 0
    eval_seconds: float = 0.0
    tokens_per_second: float = 0.0
    response_preview: str = ""
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def add_ollama_metrics(self, final_chunk) -> None:
        """Copy the counters Ollama reports on the last streamed chunk (durations are in ns)."""
        self.load_seconds = (final_chunk.load_duration or 0) / NS
        self.prompt_tokens = final_chunk.prompt_eval_count or 0
        self.prompt_eval_seconds = (final_chunk.prompt_eval_duration or 0) / NS
        self.output_tokens = final_chunk.eval_count or 0
        self.eval_seconds = (final_chunk.eval_duration or 0) / NS
        if self.eval_seconds:
            self.tokens_per_second = self.output_tokens / self.eval_seconds

    def summary_line(self) -> str:
        parts = [f"{self.wall_seconds:.2f} s total"]
        if self.stt_seconds is not None:
            parts.append(f"Whisper {self.stt_seconds:.2f} s")
        parts += [
            f"prompt {self.prompt_tokens} tok",
            f"output {self.output_tokens} tok",
            f"{self.tokens_per_second:.1f} tok/s",
        ]
        if self.load_seconds >= 0.5:
            parts.append(f"model load {self.load_seconds:.1f} s")
        return " · ".join(parts)


class InferenceTracker:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, record: InferenceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock, self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> dict:
        records = [r for r in self.records() if not r.get("error")]
        by_modality: dict[str, int] = {}
        for r in records:
            key = " + ".join(r["modalities"])
            by_modality[key] = by_modality.get(key, 0) + 1
        speeds = [r["tokens_per_second"] for r in records if r["tokens_per_second"]]
        return {
            "requests": len(records),
            "by_modality": by_modality,
            "avg_tokens_per_second": sum(speeds) / len(speeds) if speeds else 0.0,
            "avg_latency_seconds": sum(r["wall_seconds"] for r in records) / len(records) if records else 0.0,
            "total_output_tokens": sum(r["output_tokens"] for r in records),
        }
