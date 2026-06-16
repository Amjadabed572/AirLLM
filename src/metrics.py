"""Performance measurement primitives (Task 5.4).

Captures every metric the assignment asks for, in raw form so the plots can be
regenerated later:
  - TTFT  (Time To First Token)         -> Prefill cost / KV-cache build
  - ITL / TPOT (Time Per Output Token)  -> Decode cost / memory-bound stream
  - Throughput (tokens/sec)
  - Peak RAM and peak VRAM
  - Wall-clock time and an estimated energy figure

We sample memory in a background thread so we catch the *peak*, not just the
end state. All numbers are stored raw (see results/*.json) per the "measure
consistently and keep raw numbers" instruction.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field

import psutil


@dataclass
class RunMetrics:
    label: str                      # e.g. "baseline" / "airllm-q4"
    model: str
    quantization: str               # "fp16" | "q8" | "q4" | "none"
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0             # time to first token
    tpot_s: float = 0.0             # mean time per output token (decode)
    total_gen_s: float = 0.0
    throughput_tok_s: float = 0.0
    peak_ram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    est_energy_wh: float = 0.0
    failed: bool = False
    failure_reason: str = ""
    per_token_latencies_s: list[float] = field(default_factory=list)

    def finalize(self) -> None:
        n = max(self.output_tokens, 1)
        decode_time = max(self.total_gen_s - self.ttft_s, 1e-9)
        self.tpot_s = decode_time / max(n - 1, 1)
        self.throughput_tok_s = self.output_tokens / max(self.total_gen_s, 1e-9)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class MemorySampler:
    """Polls RSS (and VRAM if torch+cuda available) on a background thread."""

    def __init__(self, interval_s: float = 0.1, avg_power_w: float = 65.0):
        self.interval_s = interval_s
        self.avg_power_w = avg_power_w  # used for the energy estimate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_ram_gb = 0.0
        self.peak_vram_gb = 0.0
        self._proc = psutil.Process()
        self._t0 = 0.0
        self._elapsed = 0.0
        self._torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / (1024**3)
            self.peak_ram_gb = max(self.peak_ram_gb, rss)
            if self._torch is not None:
                vram = self._torch.cuda.max_memory_allocated() / (1024**3)
                self.peak_vram_gb = max(self.peak_vram_gb, vram)
            time.sleep(self.interval_s)

    def __enter__(self) -> MemorySampler:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._elapsed = time.perf_counter() - self._t0

    @property
    def energy_wh(self) -> float:
        # Coarse estimate: average package power * time. Document the assumed
        # wattage in the report; for a precise figure use a wall meter / RAPL.
        return self.avg_power_w * (self._elapsed / 3600.0)


def time_streaming_generation(token_iter, metrics: RunMetrics) -> str:
    """Consume a streaming token iterator, recording TTFT and per-token gaps.

    `token_iter` yields decoded text chunks (e.g. transformers TextIteratorStreamer).
    Returns the full generated text.
    """
    pieces: list[str] = []
    start = time.perf_counter()
    last = start
    first_seen = False
    for chunk in token_iter:
        now = time.perf_counter()
        if not first_seen:
            metrics.ttft_s = now - start
            first_seen = True
        else:
            metrics.per_token_latencies_s.append(now - last)
        last = now
        pieces.append(chunk)
        metrics.output_tokens += 1
    metrics.total_gen_s = time.perf_counter() - start
    metrics.finalize()
    return "".join(pieces)
